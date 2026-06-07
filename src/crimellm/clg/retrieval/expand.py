"""Graph traversal: expand each seed by one or two hops into related entities.

Every hop is filtered by jurisdiction (when the query carries one) and by
temporal validity (Provision versions only matter at the ``as_of`` date).
Filtering at the Cypher boundary keeps memory bounded for large graphs.

The traversal rules below are deliberately small for Phase 4 — we cover the
relations the gate query exercises (cited/citing, INTERPRETS, PART_OF
versions). Phase 5 layers ``treatment``-aware traversal on top.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date as _date
from typing import Any

from ..graph.driver import Neo4jStore, get_store
from .parse_query import Query
from .seed import Candidate


def _date_iso(d: Any) -> str | None:
    if d is None:
        return None
    if isinstance(d, _date):
        return d.isoformat()
    s = str(d)
    return s if s else None


def _candidate_from_row(row: dict[str, Any], source: str, base: float) -> Candidate:
    return Candidate(
        chunk_id=None,
        text=(row.get("text") or ""),
        parent_type=row["parent_type"],
        parent_id=row["parent_id"],
        parent_name=row.get("parent_name") or row["parent_id"],
        parent_jurisdiction=row.get("parent_jurisdiction"),
        section_path=row.get("section_path"),
        version_id=row.get("version_id"),
        decision_date=row.get("decision_date"),
        source=source,
        base_score=base,
        extras={"via": row.get("via")},
    )


# --- Case neighbours -------------------------------------------------------


def expand_case_citations(
    case_ids: Iterable[str],
    *,
    query: Query,
    limit_each: int = 5,
    store: Neo4jStore | None = None,
) -> list[Candidate]:
    """Neighbour Cases on the citation graph (both directions)."""
    store = store or get_store()
    ids = [i for i in case_ids if i]
    if not ids:
        return []
    out: list[Candidate] = []
    rows = store.run(
        """
        UNWIND $ids AS seed_id
        MATCH (seed:Case {id: seed_id})-[:CITES]->(b:Case)
        WHERE ($j IS NULL OR b.jurisdiction = $j)
        WITH seed_id, b LIMIT $limit
        RETURN 'Case' AS parent_type, b.id AS parent_id,
               b.name AS parent_name, b.jurisdiction AS parent_jurisdiction,
               b.decision_date AS decision_date,
               '' AS text, '' AS section_path, '' AS version_id,
               seed_id AS via
        """,
        ids=ids,
        j=query.jurisdiction,
        limit=limit_each * max(len(ids), 1),
    )
    for r in rows:
        out.append(_candidate_from_row(r, source="cites", base=0.55))

    rows = store.run(
        """
        UNWIND $ids AS seed_id
        MATCH (a:Case)-[:CITES]->(seed:Case {id: seed_id})
        WHERE ($j IS NULL OR a.jurisdiction = $j)
        WITH seed_id, a LIMIT $limit
        RETURN 'Case' AS parent_type, a.id AS parent_id,
               a.name AS parent_name, a.jurisdiction AS parent_jurisdiction,
               a.decision_date AS decision_date,
               '' AS text, '' AS section_path, '' AS version_id,
               seed_id AS via
        """,
        ids=ids,
        j=query.jurisdiction,
        limit=limit_each * max(len(ids), 1),
    )
    for r in rows:
        out.append(_candidate_from_row(r, source="cited_by", base=0.55))
    return out


def expand_case_interprets(
    case_ids: Iterable[str],
    *,
    query: Query,
    store: Neo4jStore | None = None,
) -> list[Candidate]:
    """For each seed Case, pull the Provisions it interprets (already date-resolved at load time)."""
    store = store or get_store()
    ids = [i for i in case_ids if i]
    if not ids:
        return []
    rows = store.run(
        """
        UNWIND $ids AS seed_id
        MATCH (:Case {id: seed_id})-[:INTERPRETS]->(p:Provision)
        WHERE ($j IS NULL OR p.jurisdiction = $j)
        RETURN 'Provision' AS parent_type, p.id AS parent_id,
               coalesce(p.section_path, p.id) AS parent_name,
               p.jurisdiction AS parent_jurisdiction,
               p.section_path AS section_path,
               p.version_id AS version_id,
               p.text AS text, NULL AS decision_date,
               seed_id AS via
        """,
        ids=ids,
        j=query.jurisdiction,
    )
    return [_candidate_from_row(r, source="interprets", base=0.6) for r in rows]


# --- Provision neighbours --------------------------------------------------


def expand_provision_as_of(
    provision_seeds: Iterable[tuple[str, str]],
    *,
    query: Query,
    store: Neo4jStore | None = None,
) -> list[Candidate]:
    """Swap each seed Provision for the version valid on ``query.as_of``.

    Input is ``(instrument_id, section_path)`` pairs — typically extracted
    from the seed candidates. The Cypher picks the most-recent version whose
    ``valid_from <= as_of`` and either ``valid_to`` is null or it's after
    ``as_of``. Skips when no version satisfies the date.
    """
    store = store or get_store()
    pairs = [{"iid": iid, "sec": sec} for iid, sec in provision_seeds if iid and sec]
    if not pairs:
        return []
    rows = store.run(
        """
        UNWIND $pairs AS p
        MATCH (pv:Provision {instrument_id: p.iid, section_path: p.sec})
        WHERE pv.valid_from IS NOT NULL AND pv.valid_from <= date($as_of)
          AND (pv.valid_to IS NULL OR date($as_of) <= pv.valid_to)
          AND ($j IS NULL OR pv.jurisdiction = $j)
        WITH p, pv ORDER BY pv.valid_from DESC
        WITH p.iid + ':' + p.sec AS k, collect(pv)[0] AS picked
        RETURN 'Provision' AS parent_type, picked.id AS parent_id,
               coalesce(picked.section_path, picked.id) AS parent_name,
               picked.jurisdiction AS parent_jurisdiction,
               picked.section_path AS section_path,
               picked.version_id AS version_id,
               picked.text AS text, NULL AS decision_date,
               k AS via
        """,
        pairs=pairs,
        as_of=_date_iso(query.as_of),
        j=query.jurisdiction,
    )
    return [_candidate_from_row(r, source="as_of", base=0.7) for r in rows]


def expand_provision_interpreting_cases(
    provision_ids: Iterable[str],
    *,
    query: Query,
    limit_each: int = 5,
    store: Neo4jStore | None = None,
) -> list[Candidate]:
    """Cases that interpret each seed Provision."""
    store = store or get_store()
    ids = [i for i in provision_ids if i]
    if not ids:
        return []
    rows = store.run(
        """
        UNWIND $ids AS pid
        MATCH (c:Case)-[:INTERPRETS]->(:Provision {id: pid})
        WHERE ($j IS NULL OR c.jurisdiction = $j)
        WITH pid, c LIMIT $limit
        RETURN 'Case' AS parent_type, c.id AS parent_id,
               c.name AS parent_name, c.jurisdiction AS parent_jurisdiction,
               c.decision_date AS decision_date,
               '' AS text, '' AS section_path, '' AS version_id,
               pid AS via
        """,
        ids=ids,
        j=query.jurisdiction,
        limit=limit_each * max(len(ids), 1),
    )
    return [_candidate_from_row(r, source="interpreted_by", base=0.55) for r in rows]


# --- Top-level expander ----------------------------------------------------


def expand_seeds(
    seeds: list[Candidate],
    *,
    query: Query,
    store: Neo4jStore | None = None,
) -> list[Candidate]:
    """Run the right expansions for whichever entity types the seeds are.

    Returns *new* candidates only (does not echo the seeds back).
    """
    case_ids = [c.parent_id for c in seeds if c.parent_type == "Case"]
    provision_ids = [c.parent_id for c in seeds if c.parent_type == "Provision"]
    provision_pairs = [
        (c.extras.get("instrument_id") or c.parent_id.rsplit("/section/", 1)[0], c.section_path)
        for c in seeds
        if c.parent_type == "Provision" and c.section_path
    ]

    out: list[Candidate] = []
    out += expand_case_citations(case_ids, query=query, store=store)
    out += expand_case_interprets(case_ids, query=query, store=store)
    out += expand_provision_as_of(provision_pairs, query=query, store=store)
    out += expand_provision_interpreting_cases(provision_ids, query=query, store=store)
    return out
