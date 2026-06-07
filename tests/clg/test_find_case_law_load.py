"""Live Neo4j gate: a UK judgment links to provisions it interprets.

The INTERPRETS edge must land on the Provision version in force on the
case's decision date. Auto-skipped when Neo4j is unreachable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.graph.loaders import (
    load_cases,
    load_instruments,
    load_interprets,
    load_provisions,
)
from crimellm.clg.graph.schema import apply_schema
from crimellm.clg.parse import find_case_law as FCL
from crimellm.clg.parse import legislation_uk as LEG

LEG_FIX = Path(__file__).parent / "fixtures" / "legislation_uk"
FCL_FIX = Path(__file__).parent / "fixtures" / "find_case_law"


@pytest.fixture
def loaded_world(neo4j_store):
    apply_schema(neo4j_store)
    with neo4j_store.session() as s:
        s.run("MATCH (n) WHERE n:Instrument OR n:Provision OR n:Case OR n:Court DETACH DELETE n")

    # Two Acts in scope: Fraud Act 2006 (2 versions) + Theft Act 1968 (enacted only).
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

    # Judgment with refs into the Fraud Act (s.2, s.3) and Theft Act (s.1).
    case, refs = FCL.parse_judgment_file(FCL_FIX / "ewca-crim-fraud-2015.xml")
    load_cases([case], store=neo4j_store)

    rows = [(case.id, case.decision_date, ref) for ref in refs]
    load_interprets(rows, store=neo4j_store)
    return neo4j_store, case, refs


def test_case_loaded(loaded_world) -> None:
    store, case, _ = loaded_world
    rows = store.run("MATCH (c:Case {id: $id}) RETURN c.name AS name", id=case.id)
    assert rows and "Smith" in rows[0]["name"]


def test_interprets_picks_enacted_version_for_2015_judgment(loaded_world) -> None:
    """Decision date 2015-06-12 is inside the 'enacted' window (2006-11-08 → 2022-12-31).

    Every INTERPRETS edge on the Fraud Act must point at the *enacted*
    Provision version, never the *current* one.
    """
    store, case, _ = loaded_world
    rows = store.run(
        "MATCH (:Case {id: $id})-[:INTERPRETS]->(p:Provision) "
        "WHERE p.instrument_id = 'uk/ukpga/2006/35' "
        "RETURN p.section_path AS section, p.version_id AS version "
        "ORDER BY section",
        id=case.id,
    )
    by_section = {r["section"]: r["version"] for r in rows}
    assert by_section.get("s.2") == "enacted"
    assert by_section.get("s.3") == "enacted"


def test_unresolved_refs_dont_create_edges(loaded_world) -> None:
    """The Theft Act ref has no matching Provision in the graph; the edge is dropped."""
    store, case, _ = loaded_world
    rows = store.run(
        "MATCH (:Case {id: $id})-[:INTERPRETS]->(p:Provision) "
        "WHERE p.instrument_id = 'uk/ukpga/1968/60' "
        "RETURN count(p) AS n",
        id=case.id,
    )
    assert rows[0]["n"] == 0


def test_interprets_edge_count_matches_resolvable_refs(loaded_world) -> None:
    store, case, _ = loaded_world
    rows = store.run(
        "MATCH (:Case {id: $id})-[r:INTERPRETS]->() RETURN count(r) AS n",
        id=case.id,
    )
    # Two resolvable refs (Fraud Act s.2 and s.3); Theft Act s.1 has no
    # Provision in the graph yet.
    assert rows[0]["n"] == 2
