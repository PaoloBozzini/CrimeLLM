"""Phase 3.4: IMPLEMENTS edge loader (Instrument → Instrument).

Round-trips two Instruments + one IMPLEMENTS edge. Auto-skips when the
isolated test Neo4j container isn't running (see ``tests/conftest.py``).
"""

from __future__ import annotations

from datetime import date

from crimellm.clg.graph import (
    apply_schema,
    load_implements,
    load_instruments,
)
from crimellm.clg.models import Instrument, Provenance


def _instrument(id_: str, jurisdiction: str, short_title: str) -> Instrument:
    return Instrument(
        id=id_,
        jurisdiction=jurisdiction,
        short_title=short_title,
        year=2016,
        provenance=[
            Provenance(
                source="test",
                source_url="https://example.org",
                retrieved_at=date(2025, 1, 1),
                source_id=id_,
            )
        ],
    )


def test_load_implements_merges_edge(neo4j_store):
    apply_schema(neo4j_store)
    with neo4j_store.session() as s:
        s.run(
            "MATCH (n) WHERE n:Instrument OR n:Provision OR n:Case OR n:Court "
            "DETACH DELETE n"
        )

    eu_dir = _instrument(
        "eu/celex/32019L0770", "EU", "Directive (EU) 2019/770 on digital content"
    )
    dk_lbk = _instrument(
        "dk/lbk/2022/1234", "DK", "Lov om aftaler om digitalt indhold"
    )
    load_instruments([eu_dir, dk_lbk], store=neo4j_store)

    n = load_implements(
        [(dk_lbk.id, eu_dir.id, "preamble §1")],
        store=neo4j_store,
    )
    assert n == 1

    rows = neo4j_store.run(
        "MATCH (src:Instrument)-[r:IMPLEMENTS]->(tgt:Instrument) "
        "RETURN src.id AS src, tgt.id AS tgt, r.raw_ref AS raw"
    )
    assert len(rows) == 1
    assert rows[0]["src"] == dk_lbk.id
    assert rows[0]["tgt"] == eu_dir.id
    assert rows[0]["raw"] == "preamble §1"


def test_load_implements_idempotent(neo4j_store):
    apply_schema(neo4j_store)
    with neo4j_store.session() as s:
        s.run(
            "MATCH (n) WHERE n:Instrument OR n:Provision OR n:Case OR n:Court "
            "DETACH DELETE n"
        )

    eu_reg = _instrument("eu/celex/32016R0679", "EU", "GDPR")
    dk = _instrument("dk/lbk/2018/502", "DK", "Databeskyttelsesloven")
    load_instruments([eu_reg, dk], store=neo4j_store)
    load_implements([(dk.id, eu_reg.id, "ref-A")], store=neo4j_store)
    load_implements([(dk.id, eu_reg.id, "ref-A-updated")], store=neo4j_store)
    rows = neo4j_store.run(
        "MATCH (src:Instrument)-[r:IMPLEMENTS]->(tgt:Instrument) "
        "RETURN count(r) AS n"
    )
    assert rows[0]["n"] == 1


def test_load_implements_skips_missing_target(neo4j_store):
    apply_schema(neo4j_store)
    with neo4j_store.session() as s:
        s.run(
            "MATCH (n) WHERE n:Instrument OR n:Provision OR n:Case OR n:Court "
            "DETACH DELETE n"
        )

    dk = _instrument("dk/lbk/2099/999", "DK", "Hypothetical DK Act")
    load_instruments([dk], store=neo4j_store)
    # Target eu/celex/32099R9999 doesn't exist — MATCH fails → row silently
    # dropped, no exception.
    n = load_implements(
        [(dk.id, "eu/celex/32099R9999", None)],
        store=neo4j_store,
    )
    assert n == 1  # batch size reported, even when edge MERGE was a no-op
    rows = neo4j_store.run(
        "MATCH ()-[r:IMPLEMENTS]->() RETURN count(r) AS n"
    )
    assert rows[0]["n"] == 0
