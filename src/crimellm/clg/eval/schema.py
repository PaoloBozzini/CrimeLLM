"""Gold-set schema + loader.

The gold set is a YAML list of ``GoldQuestion`` records. Authored by hand,
small (Phase 6 target: 25-40 questions), balanced across US/UK and across
four task types: single-fact lookup, multi-hop traversal, as-of-date,
good-law check.

Each question records its expected answer as a set of authority identifiers
(``cl-cluster-...``, ``uk/ukpga/.../section/...@version`` etc.) — never
free-text. That's what makes citation accuracy a deterministic check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml

from ..models import Jurisdiction

TaskType = Literal[
    "single_fact",
    "multi_hop",
    "as_of_date",
    "good_law",
    "no_fabrication",
]

__all__ = ["TaskType", "Jurisdiction", "GoldQuestion", "GoldSet", "load_gold_set"]


@dataclass(slots=True)
class GoldQuestion:
    """One annotated question.

    ``expected_authorities`` is the set of identifiers the retrieval pipeline
    must surface — used for recall@k and citation accuracy. Order doesn't
    matter; we treat it as a set.

    ``expected_good_law`` maps a target case id to the label the cascade
    should ultimately produce (e.g. ``{"cl-cluster-plessy": "overruled"}``).
    Empty when the question doesn't probe good-law behaviour.

    ``expected_treating_case`` names the case responsible for the adverse
    treatment. When set, we additionally require the answer text to mention
    it by id — that's the "names the overruling case" gate.
    """

    id: str
    question: str
    task_type: TaskType
    jurisdiction: Jurisdiction | None = None
    as_of: date | None = None
    expected_authorities: list[str] = field(default_factory=list)
    expected_good_law: dict[str, str] = field(default_factory=dict)
    expected_treating_case: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class GoldSet:
    """A collection of ``GoldQuestion`` records + tiny metadata."""

    name: str
    description: str
    questions: list[GoldQuestion] = field(default_factory=list)
    version: str = "1"

    def __iter__(self):
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)

    def filter_by_jurisdiction(self, codes: list[str]) -> GoldSet:
        """Return a new ``GoldSet`` keeping only questions whose
        ``jurisdiction`` is in ``codes`` (case-insensitive).

        Cross-jurisdiction questions (``jurisdiction: null``) are
        included when ``codes`` contains the literal ``"XJ"`` token or
        when the caller explicitly listed any of the special tokens
        ``"ALL"`` / ``"*"`` / ``"NULL"`` — that way an operator can
        ``clg eval --jurisdiction DK,XJ`` to score DK questions plus
        cross-jurisdiction IMPLEMENTS-edge probes side-by-side.
        """
        wanted = {c.strip().upper() for c in codes if c.strip()}
        include_null = bool(wanted & {"XJ", "NULL", "ALL", "*"})
        if "ALL" in wanted or "*" in wanted:
            return self
        kept = [
            q
            for q in self.questions
            if (q.jurisdiction is None and include_null)
            or (q.jurisdiction is not None and q.jurisdiction.upper() in wanted)
        ]
        return GoldSet(
            name=self.name,
            description=self.description,
            questions=kept,
            version=self.version,
        )

    def filter_by_task_type(self, task_types: list[str]) -> GoldSet:
        """Return a new ``GoldSet`` keeping only the listed task types."""
        wanted = {t.strip().lower() for t in task_types if t.strip()}
        kept = [q for q in self.questions if q.task_type.lower() in wanted]
        return GoldSet(
            name=self.name,
            description=self.description,
            questions=kept,
            version=self.version,
        )

    def jurisdictions(self) -> list[str]:
        """Distinct jurisdiction codes present (``"XJ"`` for cross-juris)."""
        out: set[str] = set()
        for q in self.questions:
            out.add(q.jurisdiction if q.jurisdiction is not None else "XJ")
        return sorted(out)


def _to_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    raise TypeError(f"unsupported as_of type {type(value).__name__}: {value!r}")


def _question_from_row(row: dict) -> GoldQuestion:
    if "id" not in row or "question" not in row or "task_type" not in row:
        raise ValueError(f"gold question missing required field (id / question / task_type): {row}")
    return GoldQuestion(
        id=str(row["id"]),
        question=str(row["question"]),
        task_type=row["task_type"],
        jurisdiction=row.get("jurisdiction"),
        as_of=_to_date(row.get("as_of")),
        expected_authorities=list(row.get("expected_authorities", []) or []),
        expected_good_law=dict(row.get("expected_good_law", {}) or {}),
        expected_treating_case=row.get("expected_treating_case"),
        tags=list(row.get("tags", []) or []),
        notes=str(row.get("notes", "") or ""),
    )


def load_gold_set(path: Path | str) -> GoldSet:
    """Read a YAML gold-set file from disk."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"top-level YAML must be a mapping; got {type(raw).__name__}")
    questions = [_question_from_row(r) for r in (raw.get("questions") or [])]
    return GoldSet(
        name=str(raw.get("name", Path(path).stem)),
        description=str(raw.get("description", "") or ""),
        version=str(raw.get("version", "1") or "1"),
        questions=questions,
    )
