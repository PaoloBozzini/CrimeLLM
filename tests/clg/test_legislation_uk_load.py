"""Live Neo4j gate: as-of two dates returns two different texts.

Auto-skipped when Neo4j is unreachable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.graph.loaders import (
    load_instruments,
    load_provisions,
    provision_as_of,
)
from crimellm.clg.graph.schema import apply_schema
from crimellm.clg.parse import legislation_uk as P

FIX = Path(__file__).parent / "fixtures" / "legislation_uk"


@pytest.fixture
def loaded_act(neo4j_store):
    apply_schema(neo4j_store)
    with neo4j_store.session() as s:
        s.run("MATCH (n) WHERE n:Instrument OR n:Provision DETACH DELETE n")

    inst_e, prov_e = P.parse_act_file(
        FIX / "ukpga-2006-35-enacted.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="enacted",
        valid_from=date(2006, 11, 8),
        valid_to=date(2022, 12, 31),
    )
    inst_c, prov_c = P.parse_act_file(
        FIX / "ukpga-2006-35-current.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="current",
        valid_from=date(2023, 1, 1),
    )
    load_instruments([inst_e, inst_c], store=neo4j_store)
    load_provisions(prov_e + prov_c, store=neo4j_store)
    return neo4j_store


def test_instrument_is_deduplicated_across_versions(loaded_act) -> None:
    rows = loaded_act.run("MATCH (i:Instrument {id: 'uk/ukpga/2006/35'}) RETURN count(i) AS n")
    assert rows[0]["n"] == 1


def test_provisions_per_version(loaded_act) -> None:
    rows = loaded_act.run(
        "MATCH (p:Provision) WHERE p.instrument_id = 'uk/ukpga/2006/35' RETURN count(p) AS n"
    )
    # 3 sections × 2 versions
    assert rows[0]["n"] == 6


def test_part_of_edges_exist(loaded_act) -> None:
    rows = loaded_act.run(
        "MATCH (p:Provision)-[:PART_OF]->(i:Instrument {id: 'uk/ukpga/2006/35'}) "
        "RETURN count(p) AS n"
    )
    assert rows[0]["n"] == 6


def test_as_of_picks_enacted_version_for_old_date(loaded_act) -> None:
    row = provision_as_of("uk/ukpga/2006/35", "s.3", "2010-06-01", store=loaded_act)
    assert row is not None
    assert row["version_id"] == "enacted"
    assert "7 years" in row["text"]


def test_as_of_picks_current_version_for_recent_date(loaded_act) -> None:
    row = provision_as_of("uk/ukpga/2006/35", "s.3", "2024-06-01", store=loaded_act)
    assert row is not None
    assert row["version_id"] == "current"
    assert "10 years" in row["text"]


def test_as_of_before_enactment_returns_none(loaded_act) -> None:
    row = provision_as_of("uk/ukpga/2006/35", "s.3", "1990-01-01", store=loaded_act)
    assert row is None


def test_gate_two_dates_two_texts(loaded_act) -> None:
    """Phase 2 gate query — explicit."""
    old = provision_as_of("uk/ukpga/2006/35", "s.3", "2010-06-01", store=loaded_act)
    new = provision_as_of("uk/ukpga/2006/35", "s.3", "2024-06-01", store=loaded_act)
    assert old is not None and new is not None
    assert old["text"] != new["text"]
    assert old["version_id"] != new["version_id"]
