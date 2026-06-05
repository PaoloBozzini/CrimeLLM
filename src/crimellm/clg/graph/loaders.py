"""Batched UNWIND MERGE loaders. Idempotent under re-runs."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Iterator

from ..models import Case, Citation, Court
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


def _date_str(d) -> str | None:
    return d.isoformat() if d is not None else None


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
            rows = [{
                "id": c.id, "jurisdiction": c.jurisdiction, "name": c.name,
                "level": c.level, "parent_id": c.parent_id,
            } for c in batch]
            s.run(_CYPHER_COURTS, rows=rows)
            total += len(rows)
    return total


# --- Case + DECIDED --------------------------------------------------------

_CYPHER_CASES = """
UNWIND $rows AS row
MATCH (j:Jurisdiction {code: row.jurisdiction})
MERGE (c:Case {id: row.id})
  ON CREATE SET c.name = row.name, c.jurisdiction = row.jurisdiction,
                c.decision_date = date(row.decision_date),
                c.court_id = row.court_id,
                c.citations = row.citations,
                c.source = row.source, c.source_url = row.source_url,
                c.source_id = row.source_id, c.retrieved_at = date(row.retrieved_at)
  ON MATCH  SET c.name = coalesce(row.name, c.name),
                c.decision_date = coalesce(date(row.decision_date), c.decision_date),
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
            rows: list[dict[str, Any]] = []
            for c in batch:
                prov = c.provenance[0] if c.provenance else None
                rows.append({
                    "id": c.id, "jurisdiction": c.jurisdiction, "name": c.name,
                    "decision_date": _date_str(c.decision_date),
                    "court_id": c.court_id or "",
                    "citations": list(c.citations),
                    "source": prov.source if prov else "",
                    "source_url": prov.source_url if prov else "",
                    "source_id": prov.source_id if prov else "",
                    "retrieved_at": _date_str(prov.retrieved_at) if prov else None,
                })
            s.run(_CYPHER_CASES, rows=rows)
            total += len(rows)
    return total


# --- CITES (Case -> Case) --------------------------------------------------

_CYPHER_CITES = """
UNWIND $rows AS row
MATCH (a:Case {id: row.citing})
MATCH (b:Case {id: row.cited})
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
            rows = [{
                "citing": c.citing_case_id,
                "cited": c.cited_case_id,
                "treatment": c.treatment,
                "weight": float(c.weight),
                "citing_sentence": c.citing_sentence,
            } for c in batch]
            s.run(_CYPHER_CITES, rows=rows)
            total += len(rows)
    return total


# --- Gate-query helpers ----------------------------------------------------

def citing_cases(case_id: str, *, limit: int = 25, store: Neo4jStore | None = None) -> list[dict[str, Any]]:
    """Cases that cite the seed (inbound CITES)."""
    store = store or get_store()
    return store.run(
        "MATCH (citing:Case)-[r:CITES]->(seed:Case {id: $id}) "
        "RETURN citing.id AS id, citing.name AS name, "
        "       citing.decision_date AS decision_date, "
        "       r.treatment AS treatment "
        "ORDER BY citing.decision_date DESC LIMIT $limit",
        id=case_id, limit=limit,
    )


def cited_cases(case_id: str, *, limit: int = 25, store: Neo4jStore | None = None) -> list[dict[str, Any]]:
    """Cases the seed cites (outbound CITES)."""
    store = store or get_store()
    return store.run(
        "MATCH (seed:Case {id: $id})-[r:CITES]->(cited:Case) "
        "RETURN cited.id AS id, cited.name AS name, "
        "       cited.decision_date AS decision_date, "
        "       r.treatment AS treatment "
        "ORDER BY cited.decision_date DESC LIMIT $limit",
        id=case_id, limit=limit,
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
