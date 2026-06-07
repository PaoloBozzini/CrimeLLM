"""Batched UNWIND MERGE loaders. Idempotent under re-runs.

Row dicts come from ``models.*.to_neo4j_props()`` (see ``clg.models``). Add a
new field to a dataclass and these loaders pick it up automatically.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from ..models import Case, Chunk, Citation, Court, Instrument, Provenance, Provision
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


# --- CITES.treatment streaming + writeback (Phase 5.3) --------------------


def iter_neutral_cites(
    *,
    only_with_sentence: bool = False,
    jurisdiction: str | None = None,
    limit: int | None = None,
    store: Neo4jStore | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream ``CITES`` edges that still need classification.

    Yields dicts shaped for ``link.treatment_base.EdgeContext`` (plus the
    ``edge_id`` Neo4j surrogate so the writeback knows which edge to update).
    """
    store = store or get_store()
    where_clauses = [
        "r.treatment IS NULL OR r.treatment = 'neutral'",
    ]
    if only_with_sentence:
        where_clauses.append("coalesce(r.citing_sentence, '') <> ''")
    if jurisdiction:
        where_clauses.append("citing.jurisdiction = $j AND cited.jurisdiction = $j")
    where = " AND ".join(where_clauses)
    cypher = (
        f"MATCH (citing:Case)-[r:CITES]->(cited:Case) WHERE {where} "
        "RETURN id(r) AS edge_id, "
        "       citing.id AS citing_case_id, citing.name AS citing_case_name, "
        "       citing.decision_date AS citing_decision_date, "
        "       cited.id AS cited_case_id, cited.name AS cited_case_name, "
        "       cited.decision_date AS cited_decision_date, "
        "       coalesce(r.citing_sentence, '') AS citing_sentence, "
        "       coalesce(r.weight, 1.0) AS depth"
    )
    if limit:
        cypher += f" LIMIT {int(limit)}"

    with store.session() as s:
        params: dict[str, Any] = {}
        if jurisdiction:
            params["j"] = jurisdiction
        for row in s.run(cypher, **params):
            yield dict(row)


_CYPHER_WRITE_TREATMENT = """
UNWIND $rows AS row
MATCH ()-[r:CITES]->()
WHERE id(r) = row.edge_id
SET r.treatment = row.treatment,
    r.treatment_source = row.treatment_source,
    r.treatment_confidence = row.treatment_confidence,
    r.treatment_updated_at = datetime()
"""


def write_treatments(
    rows: Iterable[dict[str, Any]],
    *,
    batch_size: int = 1000,
    store: Neo4jStore | None = None,
) -> int:
    """Persist cascade results back onto each ``CITES`` edge.

    Each row is ``{edge_id, treatment, treatment_source, treatment_confidence}``.
    The cascade orchestrator's output maps cleanly onto this shape.
    """
    store = store or get_store()
    total = 0
    with store.session() as s:
        for batch in _chunks(rows, batch_size):
            payload = list(batch)
            if not payload:
                continue
            s.run(_CYPHER_WRITE_TREATMENT, rows=payload)
            total += len(payload)
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


# --- Chunk (+ PART_OF + MENTIONS) ------------------------------------------

# Chunks live in the lexical layer. PART_OF points up to the entity that
# owns the passage (Case or Provision). Embeddings live on Chunk.embedding
# so the vector index can search them directly.
_CYPHER_CHUNKS = """
UNWIND $rows AS row
MERGE (ch:Chunk {id: row.id})
  ON CREATE SET ch.text = row.text,
                ch.parent_id = row.parent_id,
                ch.parent_type = row.parent_type,
                ch.embedding_model = row.embedding_model,
                ch.embedding = row.embedding
  ON MATCH  SET ch.text = row.text,
                ch.embedding_model = row.embedding_model,
                ch.embedding = row.embedding
WITH ch, row
CALL (ch, row) {
  WITH ch, row
  WHERE row.parent_type = 'Case'
  MATCH (e:Case {id: row.parent_id})
  MERGE (ch)-[:PART_OF]->(e)
}
CALL (ch, row) {
  WITH ch, row
  WHERE row.parent_type = 'Provision'
  MATCH (e:Provision {id: row.parent_id})
  MERGE (ch)-[:PART_OF]->(e)
}
"""


def load_chunks(
    chunks: Iterable[Chunk],
    *,
    embedding_model: str,
    batch_size: int = 256,
    store: Neo4jStore | None = None,
) -> int:
    """MERGE Chunk nodes + ``PART_OF`` edges in batches.

    Chunks must already carry an ``embedding`` list of floats. Use the
    Embedder + chunker upstream to populate it; this loader is the dumb
    sink. ``embedding_model`` is stored on each Chunk so we can audit which
    backend produced which vector.
    """
    store = store or get_store()
    total = 0
    with store.session() as s:
        for batch in _chunks(chunks, batch_size):
            rows = []
            for ch in batch:
                if ch.embedding is None:
                    raise ValueError(
                        f"chunk {ch.id} has no embedding; embed before calling load_chunks"
                    )
                rows.append(
                    {
                        "id": ch.id,
                        "text": ch.text,
                        "parent_id": ch.parent_id,
                        "parent_type": ch.parent_type,
                        "embedding_model": embedding_model,
                        "embedding": ch.embedding,
                    }
                )
            s.run(_CYPHER_CHUNKS, rows=rows)
            total += len(rows)
    return total


# --- Vector search ---------------------------------------------------------


def vector_index_dim(store: Neo4jStore | None = None) -> int | None:
    """Return the ``chunk_embedding`` vector index's configured dimension.

    None when the index is missing or doesn't expose its config (older Neo4j
    versions). Used to surface dim mismatches with a friendly error before
    Neo4j blows up mid-query.
    """
    store = store or get_store()
    rows = store.run(
        "SHOW INDEXES YIELD name, options WHERE name = 'chunk_embedding' RETURN options AS options"
    )
    if not rows:
        return None
    opts = rows[0].get("options") or {}
    cfg = opts.get("indexConfig") or {}
    raw = cfg.get("vector.dimensions")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def search_chunks(
    query_vector: Sequence[float],
    *,
    k: int = 5,
    jurisdiction: str | None = None,
    parent_type: str | None = None,
    store: Neo4jStore | None = None,
) -> list[dict[str, Any]]:
    """Top-k vector search over Chunk embeddings, resolved up to parent entity.

    Returns one row per hit with: ``chunk_id``, ``score``, ``text``,
    ``parent_type``, ``parent_id``, ``parent_name``, ``parent_jurisdiction``,
    and (for Provision parents) ``section_path`` + ``version_id``.

    Optional filters keep retrieval honest:
      * ``jurisdiction`` — drop hits whose parent isn't in that jurisdiction.
      * ``parent_type`` — restrict to ``Case`` or ``Provision``.

    Raises ``ValueError`` up front when the query vector's dimension does
    not match the ``chunk_embedding`` index — saves the user from reading
    a Neo4j ``ProcedureCallFailed`` stack trace.
    """
    store = store or get_store()
    expected_dim = vector_index_dim(store)
    if expected_dim is not None and len(query_vector) != expected_dim:
        raise ValueError(
            f"query vector has {len(query_vector)} dimensions but the "
            f"chunk_embedding index is {expected_dim}-dim. "
            "Use --backend / --model to match (e.g. --backend st for "
            "all-MiniLM-L6-v2 at 384-dim), or run "
            "`clg graph rebuild-vector-index --dim <N>` to retarget the index."
        )
    rows = store.run(
        """
        CALL db.index.vector.queryNodes('chunk_embedding', $k, $vec)
        YIELD node, score
        MATCH (node)-[:PART_OF]->(parent)
        WITH node, score, parent, labels(parent) AS lbls
        WHERE ($parent_type IS NULL OR $parent_type IN lbls)
          AND ($jurisdiction IS NULL OR parent.jurisdiction = $jurisdiction)
        RETURN node.id AS chunk_id,
               score,
               node.text AS text,
               head(lbls) AS parent_type,
               parent.id AS parent_id,
               parent.jurisdiction AS parent_jurisdiction,
               coalesce(parent.name, parent.short_title, parent.section_path) AS parent_name,
               parent.section_path AS section_path,
               parent.version_id AS version_id,
               parent.decision_date AS decision_date
        ORDER BY score DESC
        """,
        k=k,
        vec=list(query_vector),
        parent_type=parent_type,
        jurisdiction=jurisdiction,
    )
    return rows


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
