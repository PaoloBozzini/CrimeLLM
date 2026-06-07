"""Rerank candidates by a weighted sum: vector + graph centrality + recency.

Deterministic + fast. Nothing here calls an LLM — the heavyweight
synthesis step downstream gets a small, ranked, deduplicated context.

Score formula (each component normalised to [0, 1]):

    final = w_vec   * base_score
          + w_graph * count_distinct_seed_neighbours
          + w_recent * recency_score(decision_date or valid_from)

``w_*`` weights are exposed so the CLI can experiment; defaults are
deliberately mild — vector similarity is the dominant signal unless the
graph offers strong neighbourhood evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date as _date
from datetime import datetime

from .good_law import GoodLawFlag
from .seed import Candidate


@dataclass(slots=True)
class RerankWeights:
    vector: float = 1.0
    graph: float = 0.4
    recency: float = 0.2


def _as_date(value: object) -> _date | None:
    if value is None:
        return None
    if isinstance(value, _date):
        return value
    s = str(value)
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _recency_score(value: object, *, today: _date) -> float:
    """Maps a date to ``[0, 1]``: today=1.0, 50 years ago≈0.05."""
    d = _as_date(value)
    if d is None:
        return 0.0
    age_days = max((today - d).days, 0)
    half_life_days = 365 * 10  # ten years
    return math.exp(-age_days * math.log(2) / half_life_days)


def _key(c: Candidate) -> tuple[str, str]:
    return (c.parent_type, c.parent_id)


def dedupe_candidates(items: Iterable[Candidate]) -> list[Candidate]:
    """Collapse duplicates by ``(parent_type, parent_id)``, keeping the best base score and union of sources."""
    by_key: dict[tuple[str, str], Candidate] = {}
    for c in items:
        k = _key(c)
        if k not in by_key:
            by_key[k] = replace(c)
            by_key[k].extras = dict(c.extras)
            by_key[k].extras.setdefault("sources", []).append(c.source)
            continue
        kept = by_key[k]
        kept.extras.setdefault("sources", []).append(c.source)
        if c.base_score > kept.base_score:
            kept.base_score = c.base_score
            kept.text = c.text or kept.text
            kept.source = c.source
    return list(by_key.values())


def rerank(
    candidates: list[Candidate],
    *,
    weights: RerankWeights | None = None,
    today: _date | None = None,
    good_law: dict[str, list[GoodLawFlag]] | None = None,
    top_k: int = 8,
) -> list[Candidate]:
    """Score + sort the (deduped) candidate list. Returns the top ``top_k``.

    Cases flagged as overruled/reversed/etc. via ``good_law`` keep their
    spot in the list but get a small score penalty so they sink in the
    ordering. Synthesis still mentions them with the adverse-treatment
    caveat — the user wants to know they exist.
    """
    weights = weights or RerankWeights()
    today = today or _date.today()
    good_law = good_law or {}

    # Graph signal = number of distinct sources that surfaced this candidate.
    pooled = dedupe_candidates(candidates)
    max_graph = max((len(c.extras.get("sources", [])) for c in pooled), default=1)

    for c in pooled:
        srcs = c.extras.get("sources", [])
        graph_norm = (len(set(srcs)) / max_graph) if max_graph else 0.0
        recency = _recency_score(c.decision_date or c.extras.get("valid_from"), today=today)
        s = weights.vector * c.base_score + weights.graph * graph_norm + weights.recency * recency
        if good_law.get(c.parent_id):
            s -= 0.15
        c.score = s

    pooled.sort(key=lambda c: c.score, reverse=True)
    return pooled[:top_k]
