"""Phase 5.2 — distillation: sample ``CITES`` edges, label with a teacher,
write a CSV that ``classifier/train.py`` can consume at ``num_labels=10``.

Workflow:

1. ``sample_edges`` pulls a (stratified-able) sample of un-classified
   edges out of Neo4j.
2. ``label_with_teacher`` runs them through a ``TreatmentClassifier``
   (typically ``ClaudeTreatmentClassifier`` — that's the "teacher" role).
3. ``write_training_csv`` serialises ``(text, label, label_str,
   confidence, teacher)`` rows; the existing
   ``classifier.data.load_dataset_from_csv`` reads it without changes.

The teacher can be Claude or, when budget is zero, a bigger Ollama model.
Quality reflects whichever one you pick.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..graph.driver import Neo4jStore, get_store
from .treatment_base import LABEL_TO_ID, EdgeContext, TreatmentClassifier, TreatmentResult


@dataclass(slots=True)
class DistillSample:
    """One labelled edge ready for the training CSV."""

    edge: EdgeContext
    label: str
    confidence: float
    teacher: str

    def to_row(self) -> dict[str, str | int | float]:
        text = (
            f"Cited: {self.edge.cited_case_name or self.edge.cited_case_id}\n"
            f"Sentence: {self.edge.citing_sentence or ''}"
        )
        return {
            "text": text,
            "label": LABEL_TO_ID[self.label],
            "label_str": self.label,
            "confidence": round(float(self.confidence), 4),
            "teacher": self.teacher,
            "citing_case_id": self.edge.citing_case_id,
            "cited_case_id": self.edge.cited_case_id,
        }


# --- sampling --------------------------------------------------------------


def sample_edges(
    *,
    n: int,
    only_with_sentence: bool = True,
    jurisdiction: str | None = None,
    store: Neo4jStore | None = None,
    seed: int = 42,
) -> Iterator[EdgeContext]:
    """Sample ``CITES`` edges that still need treatment classification.

    Strategy: pull a candidate pool of up to ``8 × n`` edges from Neo4j
    using ``rand()`` ordering, then shuffle locally with ``seed`` and emit
    the first ``n``. The pool oversample keeps the local shuffle meaningful
    even when ``n`` is small; ``rand()`` keeps the DB-side selection fast.
    """
    store = store or get_store()
    where = [
        "(r.treatment IS NULL OR r.treatment = 'neutral')",
    ]
    if only_with_sentence:
        where.append("coalesce(r.citing_sentence, '') <> ''")
    if jurisdiction:
        where.append("citing.jurisdiction = $j AND cited.jurisdiction = $j")
    cypher = (
        "MATCH (citing:Case)-[r:CITES]->(cited:Case) "
        f"WHERE {' AND '.join(where)} "
        "WITH r, citing, cited ORDER BY rand() LIMIT $pool "
        "RETURN id(r) AS edge_id, "
        "       citing.id AS citing_case_id, citing.name AS citing_case_name, "
        "       citing.decision_date AS citing_decision_date, "
        "       cited.id AS cited_case_id, cited.name AS cited_case_name, "
        "       cited.decision_date AS cited_decision_date, "
        "       coalesce(r.citing_sentence, '') AS citing_sentence, "
        "       coalesce(r.weight, 1.0) AS depth"
    )
    pool_size = max(n * 8, n + 16)
    params: dict[str, Any] = {"pool": pool_size}
    if jurisdiction:
        params["j"] = jurisdiction

    with store.session() as s:
        rows = [dict(row) for row in s.run(cypher, **params)]

    rng = random.Random(seed)
    rng.shuffle(rows)
    for r in rows[:n]:
        yield EdgeContext(
            citing_case_id=r["citing_case_id"],
            cited_case_id=r["cited_case_id"],
            citing_sentence=r.get("citing_sentence", "") or "",
            citing_case_name=r.get("citing_case_name", "") or "",
            cited_case_name=r.get("cited_case_name", "") or "",
            citing_decision_date=str(r.get("citing_decision_date"))
            if r.get("citing_decision_date")
            else None,
            cited_decision_date=str(r.get("cited_decision_date"))
            if r.get("cited_decision_date")
            else None,
            depth=float(r.get("depth") or 1.0),
        )


# --- teacher labelling -----------------------------------------------------


def label_with_teacher(
    edges: Sequence[EdgeContext],
    *,
    teacher: TreatmentClassifier,
    batch_size: int = 16,
) -> list[DistillSample]:
    """Run a ``TreatmentClassifier`` over the edges. Returns labelled samples.

    Edges where the teacher abstains (``None``) are skipped — they don't
    add useful signal to the training set.
    """
    samples: list[DistillSample] = []
    for start in range(0, len(edges), batch_size):
        chunk = edges[start : start + batch_size]
        results: list[TreatmentResult | None] = teacher.classify_batch(chunk)
        for edge, res in zip(chunk, results, strict=True):
            if res is None:
                continue
            samples.append(
                DistillSample(
                    edge=edge,
                    label=res.label,
                    confidence=res.confidence,
                    teacher=teacher.name,
                )
            )
    return samples


# --- CSV output ------------------------------------------------------------


COLUMNS = [
    "text",
    "label",
    "label_str",
    "confidence",
    "teacher",
    "citing_case_id",
    "cited_case_id",
]


def write_training_csv(
    samples: Iterable[DistillSample],
    out_path: str | Path,
) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for s in samples:
            w.writerow(s.to_row())
            n += 1
    return n


# --- label distribution helper -------------------------------------------


def label_distribution(samples: Iterable[DistillSample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        out[s.label] = out.get(s.label, 0) + 1
    return out
