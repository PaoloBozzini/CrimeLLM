"""Phase 11.4: end-to-end removal-mechanism smoke against a live Neo4j.

Sequence:

1. Enable all five jurisdictions, apply schema, load mixed US + DK
   Instruments / Provisions / Chunks + embeddings.
2. Vector-search with no jurisdiction filter → both US and DK hits.
3. Flip ``ENABLED_JURISDICTIONS=DK,EU`` and re-run schema.
4. Confirm US Jurisdiction node + US Instruments + US Chunks **still on
   disk** (no data deletion).
5. Re-run vector search with the new settings → US hits silenced, DK hits
   intact.
6. Operator override ``--jurisdiction US`` still surfaces the US data
   (caller-knows-best, Phase 7 invariant).

Auto-skips when the test Neo4j container isn't running (see
``tests/conftest.py``).
"""

from __future__ import annotations

from datetime import date

import pytest

from crimellm.clg.config import Settings
from crimellm.clg.embed.embedder import FakeEmbedder
from crimellm.clg.graph import (
    apply_schema,
    load_chunks,
    load_instruments,
    load_provisions,
)
from crimellm.clg.graph.loaders import search_chunks
from crimellm.clg.models import Chunk, Instrument, Provenance, Provision


def _instrument(id_: str, jurisdiction: str, short_title: str) -> Instrument:
    return Instrument(
        id=id_,
        jurisdiction=jurisdiction,
        short_title=short_title,
        year=2020,
        provenance=[
            Provenance(
                source="test",
                source_url="https://example.org",
                retrieved_at=date(2025, 1, 1),
                source_id=id_,
            )
        ],
    )


def _provision(id_: str, jurisdiction: str, instrument_id: str, text: str) -> Provision:
    return Provision(
        id=id_,
        instrument_id=instrument_id,
        jurisdiction=jurisdiction,
        section_path="s.1",
        text=text,
        valid_from=date(2020, 1, 1),
        valid_to=None,
        version_id="enacted",
    )


def _chunk(chunk_id: str, parent_id: str, embedder: FakeEmbedder) -> Chunk:
    text = f"chunk body for {parent_id}"
    return Chunk(
        id=chunk_id,
        text=text,
        parent_id=parent_id,
        parent_type="Provision",
        embedding=embedder.embed(text),
    )


def _wipe(store):
    """Remove data nodes but keep the schema in place."""
    with store.session() as s:
        s.run(
            "MATCH (n) WHERE n:Instrument OR n:Provision OR n:Case OR n:Court "
            "OR n:Chunk DETACH DELETE n"
        )


def test_disabled_jurisdiction_data_preserved_but_silenced(
    neo4j_store, monkeypatch
):
    # --- step 1: enable everything, populate mixed-jurisdiction data ---
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "US,EW,UK,EU,DK")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    neo4j_store.settings = _config.get_settings()

    embedder = FakeEmbedder(dim=neo4j_store.settings.embedding_dim)
    apply_schema(neo4j_store)
    _wipe(neo4j_store)

    us_inst = _instrument("uk/x-test", "US", "Test US Act")
    dk_inst = _instrument("dk/lbk/2099/1", "DK", "Test DK Lbk")
    us_prov = _provision("us-prov-1", "US", us_inst.id, "US provision text.")
    dk_prov = _provision("dk-prov-1", "DK", dk_inst.id, "DK provision text.")

    load_instruments([us_inst, dk_inst], store=neo4j_store)
    load_provisions([us_prov, dk_prov], store=neo4j_store)
    load_chunks(
        [
            _chunk("ch-us-1", us_prov.id, embedder),
            _chunk("ch-dk-1", dk_prov.id, embedder),
        ],
        embedding_model=embedder.name,
        store=neo4j_store,
    )

    # Sanity: both jurisdictions reachable when filter is open.
    all_hits = search_chunks(
        embedder.embed("anything"),
        k=10,
        enabled_jurisdictions=None,
        store=neo4j_store,
    )
    juris = {r["parent_jurisdiction"] for r in all_hits}
    assert "US" in juris and "DK" in juris

    # --- step 3: flip enabled set ---
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK,EU")
    _config.get_settings.cache_clear()
    neo4j_store.settings = _config.get_settings()

    apply_schema(neo4j_store)  # idempotent re-run with the new enabled set

    # --- step 4: US data still on disk ---
    rows = neo4j_store.run(
        "MATCH (n) WHERE n.jurisdiction = 'US' RETURN labels(n) AS lbls, n.id AS id"
    )
    us_node_ids = {r["id"] for r in rows}
    assert us_inst.id in us_node_ids
    assert us_prov.id in us_node_ids
    # The US Jurisdiction node should also still exist (not destroyed).
    jur_rows = neo4j_store.run(
        "MATCH (j:Jurisdiction {code: 'US'}) RETURN j.code AS c"
    )
    assert any(r["c"] == "US" for r in jur_rows)

    # --- step 5: vector search now excludes US ---
    enabled = list(_config.get_settings().enabled_jurisdictions)
    hits_after = search_chunks(
        embedder.embed("anything"),
        k=10,
        enabled_jurisdictions=enabled,
        store=neo4j_store,
    )
    juris_after = {r["parent_jurisdiction"] for r in hits_after}
    assert "US" not in juris_after
    assert "DK" in juris_after

    # --- step 6: caller-knows-best override still surfaces US ---
    hits_override = search_chunks(
        embedder.embed("anything"),
        k=10,
        jurisdiction="US",
        enabled_jurisdictions=enabled,
        store=neo4j_store,
    )
    assert any(r["parent_jurisdiction"] == "US" for r in hits_override)


def test_re_enable_restores_visibility(neo4j_store, monkeypatch):
    """Flip OFF then ON — the data is reachable again without a re-ingest."""
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "US,DK")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    neo4j_store.settings = _config.get_settings()

    embedder = FakeEmbedder(dim=neo4j_store.settings.embedding_dim)
    apply_schema(neo4j_store)
    _wipe(neo4j_store)

    us_inst = _instrument("uk/x-test-2", "US", "Test US Act 2")
    us_prov = _provision("us-prov-2", "US", us_inst.id, "US provision text 2.")
    load_instruments([us_inst], store=neo4j_store)
    load_provisions([us_prov], store=neo4j_store)
    load_chunks(
        [_chunk("ch-us-2", us_prov.id, embedder)],
        embedding_model=embedder.name,
        store=neo4j_store,
    )

    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK")
    _config.get_settings.cache_clear()
    neo4j_store.settings = _config.get_settings()
    hits_disabled = search_chunks(
        embedder.embed("anything"),
        k=5,
        enabled_jurisdictions=list(_config.get_settings().enabled_jurisdictions),
        store=neo4j_store,
    )
    assert not any(r["parent_jurisdiction"] == "US" for r in hits_disabled)

    monkeypatch.setenv("ENABLED_JURISDICTIONS", "US,DK")
    _config.get_settings.cache_clear()
    neo4j_store.settings = _config.get_settings()
    hits_re_enabled = search_chunks(
        embedder.embed("anything"),
        k=5,
        enabled_jurisdictions=list(_config.get_settings().enabled_jurisdictions),
        store=neo4j_store,
    )
    assert any(r["parent_jurisdiction"] == "US" for r in hits_re_enabled)


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    yield
    _config.get_settings.cache_clear()
