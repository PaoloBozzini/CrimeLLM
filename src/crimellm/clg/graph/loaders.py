"""Batched UNWIND MERGE loaders. Idempotent under re-runs.

Row dicts come from ``models.*.to_neo4j_props()`` (see ``clg.models``). Add a
new field to a dataclass and these loaders pick it up automatically.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from ..models import Case, Citation, Court, Instrument, Provenance, Provision
from .driver import Neo4jStore, get_store


def _chunks(it: Iterable[Any], n: int) -> Iterator[list[Any]]:
    buf: list[Any] = []
    for item in it:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _merge_provenance(base: dict[str, Any], prov: Provenance | None) -> dict[str, Any]:
    """Flatten primary Provenance into the entity row.

    Provenance owns ``source / source_url / source_id / retrieved_at``. We
    denormalise onto the entity node for cheap audit queries; deep historical
    provenance lists would live on dedicated nodes if/when we need them (not
    in Phase 1).
    """
    if prov is None:
        for k in ("source", "source_url", "source_id"):
            base.setdefault(k, "")
        base.setdefault("retrieved_at", None)
        return base
    base.update(prov.to_neo4j_props())
    return base


# --- Court -----------------------------------------------------------------

_CYPHER_COURTS = """
UNWIND $rows AS row
MATCH (j:Jurisdiction {code: row.jurisdiction})
MERGE (c:Court {id: row.id})
  ON CREATE SET c.name = row.name, c.level = row.level,
                c.jurisdiction = row.jurisdiction,
                c.parent_id = row.parent_id
  ON MATCH  SET c.name = row.name, c.level = row.level,
                c.parent_id = row.parent_id
MERGE (c)-[:IN_JURISDICTION]->(j)
"""


def load_courts(
    courts: Iterable[Court],
    *,
    batch_size: int = 1000,
    store: Neo4jStore | None = None,
) -> int:
    store = store or get_store()
    total = 0
    with store.session() as s:
        for batch in _chunks(courts, batch_size):
            rows = [c.to_neo4j_props() for c in batch]
            s.run(_CYPHER_COURTS, rows=rows)
            total += len(rows)
    return total


# --- Case + DECIDED --------------------------------------------------------

_CYPHER_CASES = """
UNWIND $rows AS row
MATCH (j:Jurisdiction {code: row.jurisdiction})
MERGE (c:Case {id: row.id})
  ON CREATE SET c.name = row.name, c.jurisdiction = row.jurisdiction,
                c.decision_date = CASE WHEN row.decision_date IS NULL
                                       THEN NULL ELSE date(row.decision_date) END,
                c.court_id = row.court_id,
                c.citations = row.citations,
                c.source = row.source, c.source_url = row.source_url,
                c.source_id = row.source_id,
                c.retrieved_at = CASE WHEN row.retrieved_at IS NULL
                                      THEN NULL ELSE date(row.retrieved_at) END
  ON MATCH  SET c.name = coalesce(row.name, c.name),
                c.decision_date = coalesce(
                    CASE WHEN row.decision_date IS NULL
                         THEN NULL ELSE date(row.decision_date) END,
                    c.decision_date
                ),
                c.court_id = coalesce(row.court_id, c.court_id)
MERGE (c)-[:IN_JURISDICTION]->(j)
WITH c, row
WHERE row.court_id <> ''
MATCH (ct:Court {id: row.court_id})
MERGE (ct)-[:DECIDED]->(c)
"""


def load_cases(
    cases: Iterable[Case],
    *,
    batch_size: int = 5000,
    store: Neo4jStore | None = None,
) -> int:
    store = store or get_store()
    total = 0
    with store.session() as s:
        for batch in _chunks(cases, batch_size):
            rows = [_merge_provenance(c.to_neo4j_props(), c.primary_provenance()) for c in batch]
            s.run(_CYPHER_CASES, rows=rows)
            total += len(rows)
    return total


# --- CITES (Case -> Case) --------------------------------------------------

_CYPHER_CITES = """
UNWIND $rows AS row
MATCH (a:Case {id: row.citing_case_id})
MATCH (b:Case {id: row.cited_case_id})
MERGE (a)-[r:CITES]->(b)
  ON CREATE SET r.treatment = row.treatment,
                r.weight = row.weight,
                r.citing_sentence = row.citing_sentence
  ON MATCH  SET r.weight = row.weight,
                r.treatment = coalesce(r.treatment, row.treatment)
"""


def load_citations(
    citations: Iterable[Citation],
    *,
    batch_size: int = 10000,
    store: Neo4jStore | None = None,
) -> int:
    store = store or get_store()
    total = 0
    with store.session() as s:
        for batch in _chunks(citations, batch_size):
            rows = [c.to_neo4j_props() for c in batch]
            s.run(_CYPHER_CITES, rows=rows)
            total += len(rows)
    return total


# --- Instrument ------------------------------------------------------------

_CYPHER_INSTRUMENTS = """
UNWIND $rows AS row
MATCH (j:Jurisdiction {code: row.jurisdiction})
MERGE (i:Instrument {id: row.id})
  ON CREATE SET i.short_title = row.short_title, i.jurisdiction = row.jurisdiction,
                i.year = row.year,
                i.source = row.source, i.source_url = row.source_url,
                i.source_id = row.source_id,
                i.retrieved_at = CASE WHEN row.retrieved_at IS NULL
                                      THEN NULL ELSE date(row.retrieved_at) END
  ON MATCH  SET i.short_title = coalesce(row.short_title, i.short_title),
                i.year = coalesce(row.year, i.year)
MERGE (i)-[:IN_JURISDICTION]->(j)
"""


def load_instruments(
    instruments: Iterable[Instrument],
    *,
    batch_size: int = 1000,
    store: Neo4jStore | None = None,
) -> int:
    store = store or get_store()
    total = 0
    with store.session() as s:
        for batch in _chunks(instruments, batch_size):
            rows = [_merge_provenance(i.to_neo4j_props(), i.primary_provenance()) for i in batch]
            s.run(_CYPHER_INSTRUMENTS, rows=rows)
            total += len(rows)
    return total


# --- Provision (+ PART_OF) -------------------------------------------------

_CYPHER_PROVISIONS = """
UNWIND $rows AS row
MATCH (j:Jurisdiction {code: row.jurisdiction})
MERGE (p:Provision {id: row.id})
  ON CREATE SET p.instrument_id = row.instrument_id,
                p.jurisdiction = row.jurisdiction,
                p.section_path = row.section_path,
                p.text = row.text,
                p.version_id = row.version_id,
                p.valid_from = CASE WHEN row.valid_from IS NULL
                                    THEN NULL ELSE date(row.valid_from) END,
                p.valid_to   = CASE WHEN row.valid_to IS NULL
                                    THEN NULL ELSE date(row.valid_to) END
  ON MATCH  SET p.text = coalesce(row.text, p.text),
                p.valid_to = CASE WHEN row.valid_to IS NULL
                                  THEN p.valid_to ELSE date(row.valid_to) END
MERGE (p)-[:IN_JURISDICTION]->(j)
WITH p, row
MATCH (i:Instrument {id: row.instrument_id})
MERGE (p)-[:PART_OF]->(i)
"""


def load_provisions(
    provisions: Iterable[Provision],
    *,
    batch_size: int = 2000,
    store: Neo4jStore | None = None,
) -> int:
    store = store or get_store()
    total = 0
    with store.session() as s:
        for batch in _chunks(provisions, batch_size):
            rows = [p.to_neo4j_props() for p in batch]
            s.run(_CYPHER_PROVISIONS, rows=rows)
            total += len(rows)
    return total


# --- INTERPRETS (Case -> Provision) ----------------------------------------

# For each (case, ref) we pick the Provision version valid on the case's
# decision date. Cases with no decision_date fall back to the latest version.
_CYPHER_INTERPRETS = """
UNWIND $rows AS row
MATCH (c:Case {id: row.case_id})
OPTIONAL MATCH (p:Provision {instrument_id: row.instrument_id, section_path: row.section_path})
WHERE row.decision_date IS NULL
   OR (
       p.valid_from IS NOT NULL AND p.valid_from <= date(row.decision_date)
       AND (p.valid_to IS NULL OR date(row.decision_date) <= p.valid_to)
   )
WITH c, row, p
ORDER BY p.valid_from DESC
WITH c, row, collect(p)[0] AS picked
WHERE picked IS NOT NULL
MERGE (c)-[r:INTERPRETS]->(picked)
  ON CREATE SET r.raw_href = row.raw_href
"""


def load_interprets(
    rows: Iterable[Any],
    *,
    batch_size: int = 2000,
    store: Neo4jStore | None = None,
) -> int:
    """MERGE ``(Case)-[:INTERPRETS]->(Provision)`` edges.

    Accepts an iterable of ``(case_id, decision_date, SectionRef)`` tuples,
    or ``(case_id, decision_date, instrument_id, section_path, raw_href)``
    tuples. Each row resolves the Provision version that was in force on the
    case's decision date.
    """
    store = store or get_store()

    def _normalise(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        if len(row) == 3:
            case_id, decision_date, ref = row
            return {
                "case_id": case_id,
                "decision_date": decision_date.isoformat() if decision_date else None,
                "instrument_id": ref.instrument_id,
                "section_path": ref.section_path,
                "raw_href": ref.raw_href,
            }
        case_id, decision_date, instrument_id, section_path, raw_href = row
        return {
            "case_id": case_id,
            "decision_date": decision_date.isoformat() if decision_date else None,
            "instrument_id": instrument_id,
            "section_path": section_path,
            "raw_href": raw_href,
        }

    total = 0
    with store.session() as s:
        for batch in _chunks(rows, batch_size):
            payload = [_normalise(r) for r in batch]
            s.run(_CYPHER_INTERPRETS, rows=payload)
            total += len(payload)
    return total


# --- Temporal as-of-date helper --------------------------------------------


def provision_as_of(
    instrument_id: str,
    section_path: str,
    as_of: str,
    *,
    store: Neo4jStore | None = None,
) -> dict[str, Any] | None:
    """Return the Provision row valid on ``as_of`` (ISO date) for the section.

    Picks the most-recent version whose ``valid_from <= as_of`` and either
    ``valid_to`` is null or ``as_of <= valid_to``. Returns None if no match.
    """
    store = store or get_store()
    rows = store.run(
        """
        MATCH (p:Provision {instrument_id: $iid, section_path: $sec})
        WHERE p.valid_from IS NOT NULL AND p.valid_from <= date($as_of)
          AND (p.valid_to IS NULL OR date($as_of) <= p.valid_to)
        RETURN p.id AS id, p.version_id AS version_id,
               p.valid_from AS valid_from, p.valid_to AS valid_to,
               p.text AS text
        ORDER BY p.valid_from DESC
        LIMIT 1
        """,
        iid=instrument_id,
        sec=section_path,
        as_of=as_of,
    )
    return rows[0] if rows else None


# --- Gate-query helpers ----------------------------------------------------


def citing_cases(
    case_id: str, *, limit: int = 25, store: Neo4jStore | None = None
) -> list[dict[str, Any]]:
    """Cases that cite the seed (inbound CITES)."""
    store = store or get_store()
    return store.run(
        "MATCH (citing:Case)-[r:CITES]->(seed:Case {id: $id}) "
        "RETURN citing.id AS id, citing.name AS name, "
        "       citing.decision_date AS decision_date, "
        "       r.treatment AS treatment "
        "ORDER BY citing.decision_date DESC LIMIT $limit",
        id=case_id,
        limit=limit,
    )


def cited_cases(
    case_id: str, *, limit: int = 25, store: Neo4jStore | None = None
) -> list[dict[str, Any]]:
    """Cases the seed cites (outbound CITES)."""
    store = store or get_store()
    return store.run(
        "MATCH (seed:Case {id: $id})-[r:CITES]->(cited:Case) "
        "RETURN cited.id AS id, cited.name AS name, "
        "       cited.decision_date AS decision_date, "
        "       r.treatment AS treatment "
        "ORDER BY cited.decision_date DESC LIMIT $limit",
        id=case_id,
        limit=limit,
    )


def citation_counts(case_id: str, *, store: Neo4jStore | None = None) -> dict[str, int]:
    store = store or get_store()
    rows = store.run(
        "MATCH (c:Case {id: $id}) "
        "OPTIONAL MATCH (c)<-[in_:CITES]-() "
        "WITH c, count(in_) AS inbound "
        "OPTIONAL MATCH (c)-[out_:CITES]->() "
        "RETURN inbound, count(out_) AS outbound",
        id=case_id,
    )
    return rows[0] if rows else {"inbound": 0, "outbound": 0}
