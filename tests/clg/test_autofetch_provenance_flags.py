"""Phase A.3: auto_ingested + validated flags on Case/Instrument/Provision.

The autofetch worker tags every node it creates with ``auto_ingested=true``
and ``validated=false``. Hand-loaded nodes carry the opposite defaults.
Downstream eval filters (Phase F) read these flags via
``coalesce(n.validated, true)`` so pre-existing nodes without the property
keep their default behaviour.

Phase A only adds:
- The dataclass fields (so ``to_neo4j_props`` carries them).
- The MERGE Cypher ``ON CREATE SET`` lines (so newly created nodes persist
  the flags).
- ``ON MATCH`` preserves whatever value is already on the node — a human
  promoting an auto-ingested doc to ``validated=true`` must not be reverted
  by a subsequent reload.
"""

from __future__ import annotations

from datetime import date

from crimellm.clg.graph import loaders as L
from crimellm.clg.models import Case, Instrument, Provision


# --- dataclass defaults ----------------------------------------------------


def test_case_default_flags() -> None:
    c = Case(
        id="ECLI:DK:HR:2020:1",
        jurisdiction="DK",
        court_id="dk-hr",
        name="X",
        decision_date=date(2020, 1, 1),
    )
    props = c.to_neo4j_props()
    assert props["auto_ingested"] is False
    assert props["validated"] is True


def test_instrument_default_flags() -> None:
    i = Instrument(id="eli/lov/2020/1", jurisdiction="DK", short_title="X")
    props = i.to_neo4j_props()
    assert props["auto_ingested"] is False
    assert props["validated"] is True


def test_provision_default_flags() -> None:
    p = Provision(
        id="eli/lov/2020/1#s1",
        instrument_id="eli/lov/2020/1",
        jurisdiction="DK",
        section_path="s.1",
        text="...",
    )
    props = p.to_neo4j_props()
    assert props["auto_ingested"] is False
    assert props["validated"] is True


def test_case_auto_ingested_override() -> None:
    c = Case(
        id="ECLI:DK:HR:2020:1",
        jurisdiction="DK",
        court_id="dk-hr",
        name="X",
        decision_date=date(2020, 1, 1),
        auto_ingested=True,
        validated=False,
    )
    props = c.to_neo4j_props()
    assert props["auto_ingested"] is True
    assert props["validated"] is False


# --- loader Cypher carries the flags ---------------------------------------
#
# We pin the exact ON CREATE / ON MATCH semantics with a string check rather
# than a live-Neo4j test (which lives in tests/clg/integration). A regression
# that drops the flag from ON CREATE, or — worse — adds it to ON MATCH and
# silently un-promotes human-validated nodes, would be caught here.


def test_cypher_cases_sets_flags_on_create_only() -> None:
    cypher = L._CYPHER_CASES
    assert "c.auto_ingested = row.auto_ingested" in cypher
    assert "c.validated = row.validated" in cypher
    on_create, _, on_match = cypher.partition("ON MATCH")
    assert "auto_ingested" in on_create
    assert "validated" in on_create
    assert "auto_ingested" not in on_match
    assert "validated" not in on_match


def test_cypher_instruments_sets_flags_on_create_only() -> None:
    cypher = L._CYPHER_INSTRUMENTS
    assert "i.auto_ingested = row.auto_ingested" in cypher
    assert "i.validated = row.validated" in cypher
    on_create, _, on_match = cypher.partition("ON MATCH")
    assert "auto_ingested" in on_create
    assert "validated" in on_create
    assert "auto_ingested" not in on_match
    assert "validated" not in on_match


def test_cypher_provisions_sets_flags_on_create_only() -> None:
    cypher = L._CYPHER_PROVISIONS
    assert "p.auto_ingested = row.auto_ingested" in cypher
    assert "p.validated = row.validated" in cypher
    on_create, _, on_match = cypher.partition("ON MATCH")
    assert "auto_ingested" in on_create
    assert "validated" in on_create
    assert "auto_ingested" not in on_match
    assert "validated" not in on_match
