"""Smoke test: schema applies idempotently against a live Neo4j (skipped otherwise)."""
from __future__ import annotations

from crimellm.clg.graph.schema import apply_schema, schema_status


def test_apply_schema_idempotent(neo4j_store) -> None:
    counts1 = apply_schema(neo4j_store)
    counts2 = apply_schema(neo4j_store)
    assert counts1 == counts2
    assert counts1["constraints"] >= 7
    assert counts1["vector_index"] == 1


def test_status_shows_constraints(neo4j_store) -> None:
    apply_schema(neo4j_store)
    status = schema_status(neo4j_store)
    assert "case_id" in status["constraints"]
    assert "chunk_embedding" in status["indexes"]


def test_jurisdictions_seeded(neo4j_store) -> None:
    apply_schema(neo4j_store)
    rows = neo4j_store.run(
        "MATCH (j:Jurisdiction) RETURN j.code AS code ORDER BY code"
    )
    codes = {r["code"] for r in rows}
    assert {"US", "EW", "UK"}.issubset(codes)
