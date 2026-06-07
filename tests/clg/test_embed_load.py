"""End-to-end embed + vector search gate (auto-skipped if Neo4j unreachable).

Loads a small Fraud Act fixture, embeds with FakeEmbedder, runs vector
search, and verifies that the top hit resolves up to its Provision parent.
This is the Phase 3 gate.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.embed.chunker import chunk_provision
from crimellm.clg.embed.embedder import FakeEmbedder
from crimellm.clg.graph.loaders import (
    load_chunks,
    load_instruments,
    load_provisions,
    search_chunks,
    vector_index_dim,
)
from crimellm.clg.graph.schema import apply_schema, rebuild_vector_index
from crimellm.clg.parse import legislation_uk as LEG

LEG_FIX = Path(__file__).parent / "fixtures" / "legislation_uk"


@pytest.fixture
def loaded_chunks(neo4j_store):
    apply_schema(neo4j_store)
    # Be self-sufficient: if a previous run (or the user's live DB) rebuilt
    # the vector index at a different dim, force it back to 1024 so this
    # test's FakeEmbedder(dim=1024) matches.
    if vector_index_dim(neo4j_store) != 1024:
        rebuild_vector_index(1024, drop_chunks=True, store=neo4j_store)
    with neo4j_store.session() as s:
        s.run("MATCH (n) WHERE n:Instrument OR n:Provision OR n:Chunk DETACH DELETE n")

    inst_e, prov_e = LEG.parse_act_file(
        LEG_FIX / "ukpga-2006-35-enacted.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="enacted",
        valid_from=date(2006, 11, 8),
    )
    load_instruments([inst_e], store=neo4j_store)
    load_provisions(prov_e, store=neo4j_store)

    embedder = FakeEmbedder(dim=1024)
    chunks = []
    for p in prov_e:
        chunks.extend(chunk_provision(p))
    vecs = embedder.embed_batch([c.text for c in chunks])
    for c, v in zip(chunks, vecs, strict=True):
        c.embedding = v
    load_chunks(chunks, embedding_model=embedder.name, store=neo4j_store)
    return neo4j_store, chunks, embedder, prov_e


def test_chunks_landed(loaded_chunks) -> None:
    store, chunks, _, _ = loaded_chunks
    rows = store.run("MATCH (ch:Chunk) RETURN count(ch) AS n")
    assert rows[0]["n"] == len(chunks)


def test_part_of_edges_resolve_to_parent(loaded_chunks) -> None:
    store, _, _, _ = loaded_chunks
    rows = store.run("MATCH (ch:Chunk)-[:PART_OF]->(p:Provision) RETURN count(ch) AS n")
    assert rows[0]["n"] > 0


def test_search_resolves_to_entity(loaded_chunks) -> None:
    """Phase 3 gate — vector search returns relevant passages + resolves up."""
    store, chunks, embedder, prov_e = loaded_chunks
    seed = next(p for p in prov_e if p.section_path == "s.3")  # penalty section
    # The chunker normalises whitespace, so query against the chunk's exact
    # indexed text — that's what FakeEmbedder needs for a perfect cosine.
    seed_chunk = next(ch for ch in chunks if ch.parent_id == seed.id)
    qvec = embedder.embed(seed_chunk.text)
    hits = search_chunks(qvec, k=3, store=store)
    assert hits, "search returned no rows"
    top = hits[0]
    # FakeEmbedder is deterministic + L2-normalised → exact-match query
    # against indexed text scores 1.0 on cosine.
    assert top["parent_type"] == "Provision"
    assert top["parent_id"] == seed.id
    assert top["section_path"] == "s.3"
    assert top["score"] > 0.99


def test_search_can_filter_by_jurisdiction(loaded_chunks) -> None:
    store, _, embedder, prov_e = loaded_chunks
    qvec = embedder.embed(prov_e[0].text)
    uk_hits = search_chunks(qvec, k=5, jurisdiction="UK", store=store)
    assert all(h["parent_jurisdiction"] == "UK" for h in uk_hits)
    us_hits = search_chunks(qvec, k=5, jurisdiction="US", store=store)
    assert us_hits == []


def test_idempotent_reload(loaded_chunks) -> None:
    store, chunks, embedder, _ = loaded_chunks
    load_chunks(chunks, embedding_model=embedder.name, store=store)
    rows = store.run("MATCH (ch:Chunk) RETURN count(ch) AS n")
    assert rows[0]["n"] == len(chunks)
