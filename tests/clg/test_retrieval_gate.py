"""Phase 4 gate — end-to-end on the Phase 2 + Phase 3 fixtures.

Loads the Fraud Act (2 versions) + the 2015 EWCA judgment, embeds Provisions
with FakeEmbedder, then runs ``run_query`` and asserts:

* the answer cites *only* identifiers from the retrieved context (no
  fabrication);
* the citations include the seed Provision the question asks about; and
* the as-of-date filter picks the right Provision version.

Auto-skipped when Neo4j is unreachable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.embed.chunker import chunk_provision
from crimellm.clg.embed.embedder import FakeEmbedder
from crimellm.clg.graph.loaders import (
    load_cases,
    load_chunks,
    load_instruments,
    load_interprets,
    load_provisions,
    vector_index_dim,
)
from crimellm.clg.graph.schema import apply_schema, rebuild_vector_index
from crimellm.clg.parse import find_case_law as FCL
from crimellm.clg.parse import legislation_uk as LEG
from crimellm.clg.retrieval import (
    FakeSynthesizer,
    check_citations,
    run_query,
)
from crimellm.clg.retrieval.synthesize import _allowed_identifiers  # noqa: E501

LEG_FIX = Path(__file__).parent / "fixtures" / "legislation_uk"
FCL_FIX = Path(__file__).parent / "fixtures" / "find_case_law"


@pytest.fixture
def loaded_world(neo4j_store):
    apply_schema(neo4j_store)
    if vector_index_dim(neo4j_store) != 1024:
        rebuild_vector_index(1024, drop_chunks=True, store=neo4j_store)
    with neo4j_store.session() as s:
        s.run("MATCH (n) WHERE n:Instrument OR n:Provision OR n:Case OR n:Chunk DETACH DELETE n")

    inst_e, prov_e = LEG.parse_act_file(
        LEG_FIX / "ukpga-2006-35-enacted.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="enacted",
        valid_from=date(2006, 11, 8),
        valid_to=date(2022, 12, 31),
    )
    inst_c, prov_c = LEG.parse_act_file(
        LEG_FIX / "ukpga-2006-35-current.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="current",
        valid_from=date(2023, 1, 1),
    )
    load_instruments([inst_e, inst_c], store=neo4j_store)
    load_provisions(prov_e + prov_c, store=neo4j_store)

    embedder = FakeEmbedder(dim=1024)
    chunks = []
    for p in prov_e + prov_c:
        chunks.extend(chunk_provision(p))
    vecs = embedder.embed_batch([c.text for c in chunks])
    for c, v in zip(chunks, vecs, strict=True):
        c.embedding = v
    load_chunks(chunks, embedding_model=embedder.name, store=neo4j_store)

    # 2015 EWCA judgment that interprets Fraud Act s.2 and s.3.
    case, refs = FCL.parse_judgment_file(FCL_FIX / "ewca-crim-fraud-2015.xml")
    load_cases([case], store=neo4j_store)
    load_interprets([(case.id, case.decision_date, r) for r in refs], store=neo4j_store)

    return neo4j_store, embedder, prov_e + prov_c, case


def test_gate_no_fabricated_citations(loaded_world) -> None:
    """Phase 4 gate — zero tolerance for fabricated citations."""
    store, embedder, _, _ = loaded_world
    answer = run_query(
        "Penalty for fraud under UK law",
        jurisdiction="UK",
        as_of="2024-06-01",
        embedder=embedder,
        synthesizer=FakeSynthesizer(),
        store=store,
    )
    allowed = _allowed_identifiers(answer.used_candidates)
    valid, fabricated = check_citations(answer.text, allowed)
    assert fabricated == [], f"fabricated citations: {fabricated}"
    assert valid, "answer cited nothing — gate requires grounded citations"


def test_gate_cites_relevant_authority(loaded_world) -> None:
    """The penalty question should retrieve the s.3 Provision and cite it by id."""
    store, embedder, provisions, _ = loaded_world
    seed = next(p for p in provisions if p.section_path == "s.3" and p.version_id == "enacted")
    chunk = chunk_provision(seed)[0]
    # Anchor the query to the indexed chunk's text so FakeEmbedder lands on it.
    answer = run_query(
        chunk.text,
        jurisdiction="UK",
        as_of="2010-06-01",
        embedder=embedder,
        synthesizer=FakeSynthesizer(),
        store=store,
    )
    assert seed.id in answer.citations


def test_as_of_filter_picks_enacted_version(loaded_world) -> None:
    """For 2010 the s.3 candidate set should include the enacted version."""
    store, embedder, _, _ = loaded_world
    answer = run_query(
        "Penalty for fraud under UK law",
        jurisdiction="UK",
        as_of="2010-06-01",
        embedder=embedder,
        synthesizer=FakeSynthesizer(),
        store=store,
        top_k=10,
    )
    s3_versions = {
        c.version_id
        for c in answer.used_candidates
        if c.parent_type == "Provision" and c.section_path == "s.3"
    }
    assert "enacted" in s3_versions


def test_judgment_interprets_provision_links(loaded_world) -> None:
    """Seeding the 2015 judgment should expand into the Fraud Act Provisions."""
    store, embedder, _, case = loaded_world
    # Embed Case body inline via the chunker is Phase 4-future; instead we
    # query for the case directly and check that expansion picked up the
    # provisions it interprets.
    from crimellm.clg.retrieval import expand_seeds
    from crimellm.clg.retrieval.parse_query import Query
    from crimellm.clg.retrieval.seed import Candidate

    seed = Candidate(
        chunk_id=None,
        text="",
        parent_type="Case",
        parent_id=case.id,
        parent_name=case.name,
        parent_jurisdiction="EW",
        source="seed",
        base_score=1.0,
    )
    expansions = expand_seeds(
        [seed],
        query=Query(raw="x", jurisdiction=None, as_of=date(2015, 6, 12)),
        store=store,
    )
    interpreted = [
        c for c in expansions if c.parent_type == "Provision" and c.source == "interprets"
    ]
    section_paths = {c.section_path for c in interpreted}
    assert {"s.2", "s.3"}.issubset(section_paths)
