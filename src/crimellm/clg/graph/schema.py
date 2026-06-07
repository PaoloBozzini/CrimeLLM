"""Idempotent constraints + vector index.

Run via `clg graph init`. Safe to re-run; uses IF NOT EXISTS everywhere.
"""

from __future__ import annotations

from .driver import Neo4jStore, get_store

CONSTRAINTS = [
    "CREATE CONSTRAINT jurisdiction_code IF NOT EXISTS FOR (j:Jurisdiction) REQUIRE j.code IS UNIQUE",
    "CREATE CONSTRAINT court_id IF NOT EXISTS FOR (c:Court) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT instrument_id IF NOT EXISTS FOR (i:Instrument) REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT provision_id IF NOT EXISTS FOR (p:Provision) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.id IS UNIQUE",
    "CREATE CONSTRAINT judge_id IF NOT EXISTS FOR (j:Judge) REQUIRE j.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX case_jurisdiction IF NOT EXISTS FOR (c:Case) ON (c.jurisdiction)",
    "CREATE INDEX case_decision_date IF NOT EXISTS FOR (c:Case) ON (c.decision_date)",
    "CREATE INDEX instrument_jurisdiction IF NOT EXISTS FOR (i:Instrument) ON (i.jurisdiction)",
    "CREATE INDEX provision_valid_from IF NOT EXISTS FOR (p:Provision) ON (p.valid_from)",
    "CREATE INDEX provision_section_path IF NOT EXISTS FOR (p:Provision) ON (p.section_path)",
    "CREATE INDEX provision_instrument IF NOT EXISTS FOR (p:Provision) ON (p.instrument_id)",
    (
        "CREATE INDEX provision_section_lookup IF NOT EXISTS "
        "FOR (p:Provision) ON (p.instrument_id, p.section_path, p.valid_from)"
    ),
]


def _vector_index_cypher(dim: int) -> str:
    return (
        "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
        "FOR (ch:Chunk) ON ch.embedding "
        "OPTIONS { indexConfig: { "
        "`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine' "
        "} }"
    )


def rebuild_vector_index(
    dim: int,
    *,
    drop_chunks: bool = False,
    store: Neo4jStore | None = None,
) -> dict[str, int]:
    """Drop + recreate ``chunk_embedding`` at a new dimension.

    Use when switching embedder backends with a different vector size — e.g.
    moving from ``voyage-law-2`` (1024) to ``all-MiniLM-L6-v2`` (384). Old
    Chunk embeddings stay on disk but the index can no longer use them; you
    almost always want to ``drop_chunks=True`` and re-embed.
    """
    store = store or get_store()
    deleted = 0
    with store.session() as s:
        s.run("DROP INDEX chunk_embedding IF EXISTS")
        if drop_chunks:
            r = s.run("MATCH (ch:Chunk) DETACH DELETE ch RETURN count(ch) AS n")
            row = r.single()
            deleted = (row["n"] if row else 0) or 0
        s.run(_vector_index_cypher(dim), dim=dim)
    return {"dim": dim, "chunks_deleted": int(deleted)}


JURISDICTION_SEEDS = [
    {"code": "US", "name": "United States"},
    {"code": "EW", "name": "England & Wales"},
    {"code": "UK", "name": "United Kingdom"},
    {"code": "EU", "name": "European Union"},
    {"code": "DK", "name": "Denmark"},
]


def apply_schema(store: Neo4jStore | None = None) -> dict[str, int]:
    store = store or get_store()
    settings = store.settings
    counts = {"constraints": 0, "indexes": 0, "vector_index": 0, "jurisdictions": 0}
    with store.session() as s:
        for stmt in CONSTRAINTS:
            s.run(stmt)
            counts["constraints"] += 1
        for stmt in INDEXES:
            s.run(stmt)
            counts["indexes"] += 1
        s.run(_vector_index_cypher(settings.embedding_dim), dim=settings.embedding_dim)
        counts["vector_index"] = 1
        for j in JURISDICTION_SEEDS:
            s.run(
                "MERGE (j:Jurisdiction {code: $code}) SET j.name = $name",
                code=j["code"],
                name=j["name"],
            )
            counts["jurisdictions"] += 1
    return counts


def drop_schema(store: Neo4jStore | None = None) -> None:
    """Drop all clg constraints + indexes. Does NOT delete data."""
    store = store or get_store()
    with store.session() as s:
        for stmt in [
            "DROP CONSTRAINT jurisdiction_code IF EXISTS",
            "DROP CONSTRAINT court_id IF EXISTS",
            "DROP CONSTRAINT case_id IF EXISTS",
            "DROP CONSTRAINT instrument_id IF EXISTS",
            "DROP CONSTRAINT provision_id IF EXISTS",
            "DROP CONSTRAINT concept_id IF EXISTS",
            "DROP CONSTRAINT chunk_id IF EXISTS",
            "DROP CONSTRAINT judge_id IF EXISTS",
            "DROP INDEX case_jurisdiction IF EXISTS",
            "DROP INDEX case_decision_date IF EXISTS",
            "DROP INDEX provision_valid_from IF EXISTS",
            "DROP INDEX provision_section_path IF EXISTS",
            "DROP INDEX chunk_embedding IF EXISTS",
        ]:
            s.run(stmt)


def schema_status(store: Neo4jStore | None = None) -> dict[str, list[str]]:
    store = store or get_store()
    with store.session() as s:
        constraints = [r["name"] for r in s.run("SHOW CONSTRAINTS YIELD name")]
        indexes = [r["name"] for r in s.run("SHOW INDEXES YIELD name")]
    return {"constraints": sorted(constraints), "indexes": sorted(indexes)}
