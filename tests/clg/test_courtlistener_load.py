"""End-to-end gate query against a live Neo4j (auto-skipped otherwise).

Wipes Cases + CITES (preserves schema), loads the tiny fixture, then exercises
the gate query: inbound/outbound CITES counts for the Chevron seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crimellm.clg.graph.loaders import (
    citation_counts,
    cited_cases,
    citing_cases,
    load_cases,
    load_citations,
    load_courts,
)
from crimellm.clg.graph.schema import apply_schema
from crimellm.clg.parse import courtlistener as P

FIX = Path(__file__).parent / "fixtures" / "courtlistener"
DATE = "2024-01-01"


def _f(name: str) -> Path:
    return FIX / f"{name}-{DATE}.csv"


@pytest.fixture
def loaded(neo4j_store):
    apply_schema(neo4j_store)
    with neo4j_store.session() as s:
        s.run("MATCH (n) WHERE n:Case OR n:Court DETACH DELETE n")
    n_courts = load_courts(P.iter_courts(_f("courts")), store=neo4j_store)
    docket_to_court = P.build_docket_to_court(_f("dockets"))
    n_cases = load_cases(
        P.iter_cases(_f("opinion-clusters"), docket_to_court=docket_to_court),
        store=neo4j_store,
    )
    op = P.build_opinion_to_cluster(_f("opinions"))
    n_cites = load_citations(P.iter_citations(_f("citations"), op), store=neo4j_store)
    return neo4j_store, {"courts": n_courts, "cases": n_cases, "cites": n_cites}


def test_loaders_return_expected_counts(loaded) -> None:
    _, counts = loaded
    assert counts == {"courts": 4, "cases": 6, "cites": 8}


def test_chevron_inbound_outbound(loaded) -> None:
    store, _ = loaded
    counts = citation_counts("cl-cluster-2001", store=store)
    assert counts["inbound"] == 5  # rows 1-5 cite Chevron
    assert counts["outbound"] == 0


def test_loper_bright_cites_chevron(loaded) -> None:
    store, _ = loaded
    out = cited_cases("cl-cluster-2002", store=store)
    ids = {r["id"] for r in out}
    assert "cl-cluster-2001" in ids
    assert "cl-cluster-2003" in ids


def test_decided_edges_exist(loaded) -> None:
    store, _ = loaded
    rows = store.run(
        "MATCH (ct:Court)-[:DECIDED]->(c:Case {id: 'cl-cluster-2001'}) RETURN ct.id AS court_id"
    )
    assert rows and rows[0]["court_id"] == "scotus"


def test_idempotent_reload(loaded) -> None:
    store, _ = loaded
    # Reload everything: counts in graph must not double.
    load_courts(P.iter_courts(_f("courts")), store=store)
    docket_to_court = P.build_docket_to_court(_f("dockets"))
    load_cases(
        P.iter_cases(_f("opinion-clusters"), docket_to_court=docket_to_court),
        store=store,
    )
    op = P.build_opinion_to_cluster(_f("opinions"))
    load_citations(P.iter_citations(_f("citations"), op), store=store)

    counts = citation_counts("cl-cluster-2001", store=store)
    assert counts["inbound"] == 5
    assert counts["outbound"] == 0
    inbound = citing_cases("cl-cluster-2001", store=store)
    assert len(inbound) == 5


def test_ingest_module_url_builder() -> None:
    """Phase 1.2 helper sanity: builds well-formed bulk URLs."""
    from crimellm.clg.ingest.courtlistener import BULK_FILES, file_url

    # The "citations" key resolves to CL's OpinionsCited dump (the edge map),
    # NOT the reporter-citation table (which is filed under reporter_citations).
    url = file_url("citations", "2024-12-31")
    assert url.endswith("citation-map-2024-12-31.csv.bz2")
    url2 = file_url("reporter_citations", "2024-12-31")
    assert url2.endswith("citations-2024-12-31.csv.bz2")
    assert set(BULK_FILES) >= {"courts", "dockets", "clusters", "opinions", "citations"}
