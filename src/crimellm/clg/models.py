"""Internal document model — jurisdiction-scoped, identifier-keyed.

Sources parse into these dataclasses; the graph loader turns them into Neo4j
nodes/edges via ``to_neo4j_props()``. Identifiers (ECLI, neutral citation,
ELI) are preferred keys.

Adding a field? Add it to the dataclass. ``to_neo4j_props()`` picks it up
automatically (it walks ``__dataclass_fields__``); the loader's UNWIND row
shape stays driven by the dataclass, not by a hand-mapped dict in
``graph/loaders.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date, datetime
from typing import Any, ClassVar, Literal

Jurisdiction = Literal["US", "EW", "UK", "EU", "DK"]

Treatment = Literal[
    # Common-law set (US/UK/EU CJEU)
    "followed",
    "applied",
    "considered",
    "distinguished",
    "doubted",
    "not_followed",
    "overruled",
    "reversed",
    "affirmed",
    "neutral",
    # Civil-law set (DK Højesteret etc.) — Højesteret doesn't formally
    # overrule prior judgments; it departs from / criticises them.
    "departed_from",
    "criticised",
]


def _serialise(value: Any) -> Any:
    """Coerce a value into something Neo4j's Python driver can accept.

    Dates/datetimes → ISO strings (loader Cypher wraps them in date()/datetime()).
    Nested dataclasses → recursive props dict.
    Lists → element-wise.
    Everything else → unchanged.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialise(v) for v in value]
    if hasattr(value, "to_neo4j_props"):
        return value.to_neo4j_props()
    return value


class _Neo4jPropsMixin:
    """Adds ``to_neo4j_props()``. Designed for ``@dataclass(slots=True)``."""

    # Subclasses override to drop fields the graph loader shouldn't surface.
    _props_exclude: ClassVar[frozenset[str]] = frozenset()

    def to_neo4j_props(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            if f.name in self._props_exclude:
                continue
            out[f.name] = _serialise(getattr(self, f.name))
        return out


@dataclass(slots=True)
class Provenance(_Neo4jPropsMixin):
    source: str  # e.g. "courtlistener", "legislation.gov.uk"
    source_url: str
    retrieved_at: date
    source_id: str  # the source's native id


@dataclass(slots=True)
class Court(_Neo4jPropsMixin):
    id: str  # slug
    jurisdiction: Jurisdiction
    name: str
    level: int  # higher = more senior; binds-downward
    parent_id: str | None = None


@dataclass(slots=True)
class Case(_Neo4jPropsMixin):
    id: str  # ECLI or neutral citation or CourtListener id
    jurisdiction: Jurisdiction
    court_id: str
    name: str
    decision_date: date | None
    citations: list[str] = field(default_factory=list)  # alt identifiers
    provenance: list[Provenance] = field(default_factory=list)
    # Autofetch provenance: worker-loaded nodes set auto_ingested=True,
    # validated=False until a human promotes them. Hand-loaded nodes keep
    # the defaults below. Eval filters use coalesce(n.validated, true).
    auto_ingested: bool = False
    validated: bool = True

    # `provenance` is a list of nested objects — keep it out of the flat
    # Neo4j row dict; the loader pulls provenance[0] into top-level fields.
    _props_exclude: ClassVar[frozenset[str]] = frozenset({"provenance"})

    def primary_provenance(self) -> Provenance | None:
        return self.provenance[0] if self.provenance else None


@dataclass(slots=True)
class Instrument(_Neo4jPropsMixin):
    id: str  # ELI or short title slug
    jurisdiction: Jurisdiction
    short_title: str
    year: int | None = None
    provenance: list[Provenance] = field(default_factory=list)
    auto_ingested: bool = False
    validated: bool = True

    _props_exclude: ClassVar[frozenset[str]] = frozenset({"provenance"})

    def primary_provenance(self) -> Provenance | None:
        return self.provenance[0] if self.provenance else None


@dataclass(slots=True)
class Provision(_Neo4jPropsMixin):
    id: str  # ELI + section path, unique per version
    instrument_id: str
    jurisdiction: Jurisdiction
    section_path: str  # e.g. "s.1", "s.1(2)(a)"
    text: str
    valid_from: date | None = None
    valid_to: date | None = None
    version_id: str | None = None
    auto_ingested: bool = False
    validated: bool = True


@dataclass(slots=True)
class Concept(_Neo4jPropsMixin):
    id: str  # jurisdiction-scoped slug
    jurisdiction: Jurisdiction
    label: str


@dataclass(slots=True)
class Chunk(_Neo4jPropsMixin):
    id: str  # content hash
    text: str
    parent_id: str  # Case/Provision node id
    parent_type: Literal["Case", "Provision"]
    embedding: list[float] | None = None


@dataclass(slots=True)
class Citation(_Neo4jPropsMixin):
    """A ``(Case)-[:CITES]->(Case)`` edge with treatment + context."""

    citing_case_id: str
    cited_case_id: str
    treatment: Treatment = "neutral"
    citing_sentence: str = ""
    weight: float = 1.0
