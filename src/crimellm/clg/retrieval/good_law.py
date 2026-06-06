"""Good-law check by walking ``CITES.treatment`` edges.

A Case is flagged when one of these treatments points at it from another
Case: ``overruled``, ``reversed``, ``not_followed``, ``doubted``. The
treating Case + treatment + edge weight come back so synthesis can name
both the case and how it was overturned.

Phase 5's cascade classifier (rules → distilled → local LLM → Claude) is
what populates the ``treatment`` property in bulk. Until then most edges
are ``"neutral"`` and this check returns empty — that's expected and the
synthesizer treats absence as "no known adverse treatment".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..graph.driver import Neo4jStore, get_store

# Treatments that flip a case's good-law status. Ordered by severity so the
# synthesizer can pick the worst when more than one edge fires.
ADVERSE_TREATMENTS = (
    "overruled",
    "reversed",
    "not_followed",
    "doubted",
)


@dataclass(slots=True)
class GoodLawFlag:
    """A single adverse treatment found pointing at a target Case."""

    case_id: str
    treatment: str
    treating_case_id: str
    treating_case_name: str
    treating_decision_date: Any = None  # str / date


def check_good_law(
    case_ids: Iterable[str],
    *,
    store: Neo4jStore | None = None,
) -> dict[str, list[GoodLawFlag]]:
    """Return ``{case_id: [GoodLawFlag, ...]}`` for every adverse edge found.

    Cases without adverse treatments don't appear in the dict (cheaper for
    callers to iterate). Phase 5 will start populating the ``treatment``
    property; today most edges are ``neutral`` and this returns ``{}``.
    """
    store = store or get_store()
    ids = [i for i in case_ids if i]
    if not ids:
        return {}
    rows = store.run(
        """
        UNWIND $ids AS target_id
        MATCH (treating:Case)-[r:CITES]->(target:Case {id: target_id})
        WHERE r.treatment IN $bad
        RETURN target_id AS case_id, r.treatment AS treatment,
               treating.id AS treating_case_id,
               treating.name AS treating_case_name,
               treating.decision_date AS treating_decision_date
        ORDER BY treating.decision_date DESC
        """,
        ids=ids,
        bad=list(ADVERSE_TREATMENTS),
    )
    out: dict[str, list[GoodLawFlag]] = {}
    for r in rows:
        out.setdefault(r["case_id"], []).append(
            GoodLawFlag(
                case_id=r["case_id"],
                treatment=r["treatment"],
                treating_case_id=r["treating_case_id"],
                treating_case_name=r["treating_case_name"] or r["treating_case_id"],
                treating_decision_date=r["treating_decision_date"],
            )
        )
    return out


def summary_label(flags: list[GoodLawFlag]) -> str:
    """One-line summary for the synthesizer prompt."""
    if not flags:
        return ""
    # Pick worst by ADVERSE_TREATMENTS ordering.
    rank = {t: i for i, t in enumerate(ADVERSE_TREATMENTS)}
    worst = min(flags, key=lambda f: rank.get(f.treatment, 99))
    return f"{worst.treatment} by {worst.treating_case_name} [{worst.treating_case_id}]"
