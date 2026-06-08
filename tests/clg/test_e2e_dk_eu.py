"""Phase 12.4: end-to-end DK + EU pipeline gate.

Wires every Phase 3-8 component together against a live test Neo4j:

1. Parse EU regulation + EU directive fixtures (Phase 3).
2. Parse DK lbk fixture; preamble extractor emits IMPLEMENTS-seed pairs
   for GDPR + the LED Directive (Phase 4).
3. Load Instruments + Provisions + IMPLEMENTS edges (Phase 3.4 loader);
   the LED-side edge silently drops because the LED Directive isn't in
   the graph -- partial-ingest tolerance is part of the contract.
4. Chunk + embed the DK lbk's provisions with FakeEmbedder.
5. Run vector search via ``seed_from_chunks`` with a DA query -> expect
   the DK Provision as the top hit.
6. End-to-end synthesis through ``FakeSynthesizer`` -> DA disclaimer +
   bracketed DK Provision id in the answer (Phase 8 language routing
   verified through the full pipeline).
7. Spot-check the IMPLEMENTS edge: DK lbk -> eu/celex/32016R0679 exists,
   DK lbk -> eu/celex/32016L0680 silently dropped because the LED
   Directive Instrument isn't in the graph (Phase 3.4 invariant).

Auto-skips when the test Neo4j container isn't running (see
``tests/conftest.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crimellm.clg.embed.chunker import chunk_provision
from crimellm.clg.embed.embedder import FakeEmbedder
from crimellm.clg.graph import (
    apply_schema,
    load_chunks,
    load_implements,
    load_instruments,
    load_provisions,
)
from crimellm.clg.graph.loaders import vector_index_dim
from crimellm.clg.graph.schema import rebuild_vector_index
from crimellm.clg.parse import eurlex as P_EU
from crimellm.clg.parse import retsinformation as P_DK
from crimellm.clg.retrieval.parse_query import parse_query
from crimellm.clg.retrieval.prompts import DISCLAIMER_DA
from crimellm.clg.retrieval.seed import seed_from_chunks
from crimellm.clg.retrieval.synthesize import FakeSynthesizer

FIX = Path(__file__).parent / "fixtures"
EU_REG = FIX / "eurlex" / "32016R0679.en.fmx4.xml"
EU_DIR = FIX / "eurlex" / "32019L0770.en.fmx4.xml"
DK_LBK = FIX / "retsinformation" / "lbk-2018-502.xml"


# --- shared fixture --------------------------------------------------------


@pytest.fixture
def loaded_dk_eu_graph(neo4j_store, monkeypatch):
    """Apply schema + load DK lbk + EU regulation + EU directive + IMPLEMENTS."""
    # Phase 11 invariant: with all five jurisdictions enabled, retrieval
    # scopes to that set. Explicit env var so the test doesn't depend on
    # whatever is in the developer's .env.
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "US,EW,UK,EU,DK")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    neo4j_store.settings = _config.get_settings()

    apply_schema(neo4j_store)
    if vector_index_dim(neo4j_store) != 1024:
        rebuild_vector_index(1024, drop_chunks=True, store=neo4j_store)
    with neo4j_store.session() as s:
        s.run(
            "MATCH (n) WHERE n:Instrument OR n:Provision OR n:Case OR n:Court "
            "OR n:Chunk DETACH DELETE n"
        )

    # --- EU side ----------------------------------------------------------
    eu_reg = P_EU.parse_regulation_file(EU_REG)
    eu_dir = P_EU.parse_regulation_file(EU_DIR)
    load_instruments(
        [eu_reg.instrument, eu_dir.instrument], store=neo4j_store
    )
    load_provisions(
        list(eu_reg.provisions) + list(eu_dir.provisions),
        store=neo4j_store,
    )

    # --- DK side ----------------------------------------------------------
    dk = P_DK.parse_statute_file(DK_LBK, doc_type="lbk", year=2018, num=502)
    load_instruments([dk.instrument], store=neo4j_store)
    load_provisions(dk.provisions, store=neo4j_store)

    # --- IMPLEMENTS edges (DK lbk -> cited EU CELEX) -----------------------
    # GDPR exists; the LED Directive 2016/680 is NOT loaded -> Phase 3.4
    # silently drops the missing-target row.
    implements_edges = [
        (dk.instrument.id, f"eu/celex/{celex}", celex)
        for celex in dk.cites_eu_celex
    ]
    load_implements(implements_edges, store=neo4j_store)

    # --- embed DK Provisions ---------------------------------------------
    embedder = FakeEmbedder(dim=1024)
    chunks = []
    for p in dk.provisions:
        chunks.extend(chunk_provision(p))
    vecs = embedder.embed_batch([c.text for c in chunks])
    for c, v in zip(chunks, vecs, strict=True):
        c.embedding = v
    load_chunks(chunks, embedding_model=embedder.name, store=neo4j_store)

    return {
        "store": neo4j_store,
        "embedder": embedder,
        "chunks": chunks,
        "dk_instrument_id": dk.instrument.id,
        "dk_provision_ids": [p.id for p in dk.provisions],
        "eu_reg_id": eu_reg.instrument.id,
        "eu_dir_id": eu_dir.instrument.id,
        "dk_cites_eu_celex": list(dk.cites_eu_celex),
    }


# --- graph shape ---------------------------------------------------------


def test_graph_has_dk_and_eu_instruments(loaded_dk_eu_graph):
    store = loaded_dk_eu_graph["store"]
    rows = store.run(
        "MATCH (i:Instrument) RETURN i.id AS id, i.jurisdiction AS j"
    )
    by_juris: dict[str, set[str]] = {}
    for r in rows:
        by_juris.setdefault(r["j"], set()).add(r["id"])
    assert loaded_dk_eu_graph["eu_reg_id"] in by_juris.get("EU", set())
    assert loaded_dk_eu_graph["eu_dir_id"] in by_juris.get("EU", set())
    assert loaded_dk_eu_graph["dk_instrument_id"] in by_juris.get("DK", set())


def test_dk_provisions_loaded(loaded_dk_eu_graph):
    store = loaded_dk_eu_graph["store"]
    rows = store.run(
        "MATCH (p:Provision) WHERE p.jurisdiction = 'DK' RETURN p.id AS id"
    )
    ids = {r["id"] for r in rows}
    for pid in loaded_dk_eu_graph["dk_provision_ids"]:
        assert pid in ids


def test_implements_edge_dk_to_gdpr(loaded_dk_eu_graph):
    """The DK lbk preamble cites GDPR -> IMPLEMENTS edge materialises."""
    store = loaded_dk_eu_graph["store"]
    rows = store.run(
        "MATCH (src:Instrument {id: $src})-[:IMPLEMENTS]->(tgt:Instrument) "
        "RETURN tgt.id AS id",
        src=loaded_dk_eu_graph["dk_instrument_id"],
    )
    targets = {r["id"] for r in rows}
    assert loaded_dk_eu_graph["eu_reg_id"] in targets


def test_implements_edge_missing_target_silently_dropped(loaded_dk_eu_graph):
    """The DK lbk also cites the LED Directive (32016L0680) which we
    didn't load -- Phase 3.4 silently drops that edge instead of
    crashing."""
    store = loaded_dk_eu_graph["store"]
    rows = store.run(
        "MATCH (src:Instrument {id: $src})-[:IMPLEMENTS]->(tgt:Instrument) "
        "RETURN tgt.id AS id",
        src=loaded_dk_eu_graph["dk_instrument_id"],
    )
    targets = {r["id"] for r in rows}
    assert "eu/celex/32016L0680" not in targets
    # But the preamble extractor did find it as a candidate.
    assert "32016L0680" in loaded_dk_eu_graph["dk_cites_eu_celex"]


# --- retrieval + synthesis ----------------------------------------------


def test_seed_returns_dk_provision(loaded_dk_eu_graph):
    """Embed a chunk's exact text + run seed_from_chunks -> the matching
    DK Provision tops the results.

    The Phase 4 fixture's DK lbk text says "databeskyttelsesforordning"
    (= GDPR in Danish), which the Phase 4.5 query parser picks up as an
    EU cue (``forordning``) and would route to EU-only retrieval — a
    correct inference for a generic question, but the wrong call when
    we know the answer lives in the DK national implementation. The
    test mirrors the realistic operator workflow: explicit
    ``--jurisdiction DK`` override (caller-knows-best).
    """
    store = loaded_dk_eu_graph["store"]
    embedder = loaded_dk_eu_graph["embedder"]
    seed_chunk = loaded_dk_eu_graph["chunks"][0]
    query = parse_query(seed_chunk.text).with_overrides(jurisdiction="DK")
    candidates = seed_from_chunks(query, embedder, k=5, store=store)
    assert candidates, "seed_from_chunks returned no candidates"
    top = candidates[0]
    assert top.parent_type == "Provision"
    assert top.parent_jurisdiction == "DK"
    assert top.parent_id == seed_chunk.parent_id
    assert top.base_score > 0.99


def test_seed_respects_enabled_jurisdictions(loaded_dk_eu_graph, monkeypatch):
    """Flip ``ENABLED_JURISDICTIONS=EU`` -> DK chunk doesn't surface even
    when its own text is the query."""
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "EU")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()

    store = loaded_dk_eu_graph["store"]
    embedder = loaded_dk_eu_graph["embedder"]
    seed_chunk = loaded_dk_eu_graph["chunks"][0]
    query = parse_query(seed_chunk.text)
    candidates = seed_from_chunks(query, embedder, k=5, store=store)
    juris = {c.parent_jurisdiction for c in candidates}
    assert "DK" not in juris


def test_e2e_da_question_yields_da_answer_with_dk_citation(
    loaded_dk_eu_graph,
):
    """Full pipeline: DA query -> DK seed -> FakeSynthesizer -> DA
    disclaimer + bracketed DK Provision id in the answer text."""
    store = loaded_dk_eu_graph["store"]
    embedder = loaded_dk_eu_graph["embedder"]
    seed_chunk = loaded_dk_eu_graph["chunks"][0]

    # Phase 7 multi-signal detector picks "da" for a query with diacritics;
    # explicit --jurisdiction DK override mirrors realistic CLI use.
    query = parse_query(
        f"Hvad regulerer denne bestemmelse om behandling? {seed_chunk.text}"
    ).with_overrides(jurisdiction="DK")
    assert query.language == "da"

    candidates = seed_from_chunks(query, embedder, k=3, store=store)
    assert candidates

    ans = FakeSynthesizer().synthesise(
        query=query, candidates=candidates, good_law={}
    )
    # Phase 8: DA disclaimer prepended.
    assert ans.text.startswith(DISCLAIMER_DA)
    # Phase 12 invariant: citation guard surfaces the DK Provision id
    # exactly as it came back from the seed step.
    dk_provision_id = candidates[0].parent_id
    assert dk_provision_id.startswith("dk/lbk/2018/502")
    assert f"[{dk_provision_id}]" in ans.text
    # Audit metadata round-trips through the answer.
    d = ans.to_dict()
    assert d["language"] == "da"
    assert d["jurisdiction"] == "DK"


# --- session-scope cache reset ------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    yield
    _config.get_settings.cache_clear()
