"""Internal document model — jurisdiction-scoped, identifier-keyed.

Sources parse into these dataclasses; the graph loader turns them into Neo4j
nodes/edges. Identifiers (ECLI, neutral citation, ELI) are preferred keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

Jurisdiction = Literal["US", "EW", "UK", "EU"]

Treatment = Literal[
    "followed", "applied", "considered", "distinguished",
    "doubted", "not_followed", "overruled", "reversed",
    "affirmed", "neutral",
]


@dataclass(slots=True)
class Provenance:
    source: str               # e.g. "courtlistener", "legislation.gov.uk"
    source_url: str
    retrieved_at: date
    source_id: str            # the source's native id


@dataclass(slots=True)
class Court:
    id: str                   # slug
    jurisdiction: Jurisdiction
    name: str
    level: int                # higher = more senior; binds-downward
    parent_id: str | None = None


@dataclass(slots=True)
class Case:
    id: str                   # ECLI or neutral citation or CourtListener id
    jurisdiction: Jurisdiction
    court_id: str
    name: str
    decision_date: date | None
    citations: list[str] = field(default_factory=list)  # alt identifiers
    provenance: list[Provenance] = field(default_factory=list)


@dataclass(slots=True)
class Instrument:
    id: str                   # ELI or short title slug
    jurisdiction: Jurisdiction
    short_title: str
    year: int | None = None
    provenance: list[Provenance] = field(default_factory=list)


@dataclass(slots=True)
class Provision:
    id: str                   # ELI + section path, unique per version
    instrument_id: str
    jurisdiction: Jurisdiction
    section_path: str         # e.g. "s.1", "s.1(2)(a)"
    text: str
    valid_from: date | None = None
    valid_to: date | None = None
    version_id: str | None = None


@dataclass(slots=True)
class Concept:
    id: str                   # jurisdiction-scoped slug
    jurisdiction: Jurisdiction
    label: str


@dataclass(slots=True)
class Chunk:
    id: str                   # content hash
    text: str
    parent_id: str            # Case/Provision node id
    parent_type: Literal["Case", "Provision"]
    embedding: list[float] | None = None


@dataclass(slots=True)
class Citation:
    """A `(Case)-[:CITES]->(Case)` edge with treatment + context."""
    citing_case_id: str
    cited_case_id: str
    treatment: Treatment = "neutral"
    citing_sentence: str = ""
    weight: float = 1.0
