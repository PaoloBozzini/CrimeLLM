"""Phase 5 gate — Plessy → Brown overruling round-trip.

Build the minimal graph (two Cases + one CITES edge with an explicit
overruling sentence), run the rules-only cascade through the same code the
CLI uses, then assert:

1. The treatment edge property flipped from neutral → ``overruled``.
2. ``check_good_law`` lights up with Plessy as the flagged case.
3. ``run_query`` surfaces a "overruled by Brown v. Board" caveat in the
   answer.

Auto-skipped when Neo4j is unreachable.
"""

from __future__ import annotations

from datetime import date

import pytest

from crimellm.clg.graph.loaders import (
    iter_neutral_cites,
    load_cases,
    load_citations,
    write_treatments,
)
from crimellm.clg.graph.schema import apply_schema
from crimellm.clg.link import (
    CascadeClassifier,
    EdgeContext,
    RuleTreatmentClassifier,
)
from crimellm.clg.models import Case, Citation, Provenance
from crimellm.clg.retrieval import FakeSynthesizer, check_good_law


@pytest.fixture
def landmark_graph(neo4j_store):
    apply_schema(neo4j_store)
    with neo4j_store.session() as s:
        s.run("MATCH (n) WHERE n:Case OR n:Chunk DETACH DELETE n")

    plessy = Case(
        id="cl-cluster-plessy",
        jurisdiction="US",
        court_id="scotus",
        name="Plessy v. Ferguson",
        decision_date=date(1896, 5, 18),
        citations=["163 U.S. 537"],
        provenance=[
            Provenance(
                source="fixture",
                source_url="",
                retrieved_at=date.today(),
                source_id="plessy",
            )
        ],
    )
    brown = Case(
        id="cl-cluster-brown",
        jurisdiction="US",
        court_id="scotus",
        name="Brown v. Board of Education",
        decision_date=date(1954, 5, 17),
        citations=["347 U.S. 483"],
        provenance=[
            Provenance(
                source="fixture",
                source_url="",
                retrieved_at=date.today(),
                source_id="brown",
            )
        ],
    )
    load_cases([plessy, brown], store=neo4j_store)

    # Brown cites Plessy with an explicit overruling sentence — the rule tier
    # should pick "overruled" with high confidence.
    cite = Citation(
        citing_case_id=brown.id,
        cited_case_id=plessy.id,
        treatment="neutral",
        citing_sentence=(
            "Plessy v. Ferguson is overruled to the extent it conflicts with this opinion."
        ),
        weight=1.0,
    )
    load_citations([cite], store=neo4j_store)
    return neo4j_store, plessy, brown


def _run_rules_pass(store) -> dict[str, int]:
    """Invoke the cascade on every still-neutral CITES edge — mirrors the CLI."""
    cascade = CascadeClassifier([(RuleTreatmentClassifier(), 0.85)])
    rows = list(iter_neutral_cites(only_with_sentence=True, store=store))
    edges = [
        EdgeContext(
            citing_case_id=r["citing_case_id"],
            cited_case_id=r["cited_case_id"],
            citing_sentence=r["citing_sentence"],
            citing_case_name=r.get("citing_case_name", ""),
            cited_case_name=r.get("cited_case_name", ""),
        )
        for r in rows
    ]
    report = cascade.classify(edges)
    write_treatments(
        [
            {
                "edge_id": rows[i]["edge_id"],
                "treatment": report.results[i].label,
                "treatment_source": report.results[i].source,
                "treatment_confidence": float(report.results[i].confidence),
            }
            for i in range(len(rows))
        ],
        store=store,
    )
    return report.by_tier()


def test_gate_rules_pass_writes_overruled_treatment(landmark_graph) -> None:
    store, plessy, brown = landmark_graph
    by_tier = _run_rules_pass(store)
    assert by_tier.get("rules", 0) == 1

    rows = store.run(
        "MATCH (a:Case {id: $brown})-[r:CITES]->(b:Case {id: $plessy}) "
        "RETURN r.treatment AS treatment, r.treatment_source AS source, "
        "       r.treatment_confidence AS confidence",
        brown=brown.id,
        plessy=plessy.id,
    )
    assert rows[0]["treatment"] == "overruled"
    assert rows[0]["source"] == "rules"
    assert rows[0]["confidence"] > 0.9


def test_gate_good_law_flags_plessy(landmark_graph) -> None:
    store, plessy, brown = landmark_graph
    _run_rules_pass(store)
    flags = check_good_law([plessy.id], store=store)
    assert plessy.id in flags
    plessy_flags = flags[plessy.id]
    assert plessy_flags[0].treatment == "overruled"
    assert plessy_flags[0].treating_case_id == brown.id
    assert "Brown" in plessy_flags[0].treating_case_name


def test_gate_query_surfaces_good_law_caveat(landmark_graph) -> None:
    """Phase 5 gate — chat surface mentions the overruling case by name."""
    store, plessy, brown = landmark_graph
    _run_rules_pass(store)

    # Seed the answer with Plessy directly. Phase 4's vector seed needs
    # chunks, which we don't index here; instead we hand Plessy to the
    # rerank+synthesize part by constructing a Candidate manually.
    from crimellm.clg.retrieval.parse_query import Query
    from crimellm.clg.retrieval.rerank import rerank
    from crimellm.clg.retrieval.seed import Candidate

    plessy_candidate = Candidate(
        chunk_id=None,
        text="Plessy v. Ferguson upheld 'separate but equal'.",
        parent_type="Case",
        parent_id=plessy.id,
        parent_name=plessy.name,
        parent_jurisdiction="US",
        source="seed",
        base_score=0.95,
    )
    flags = check_good_law([plessy.id], store=store)
    ranked = rerank([plessy_candidate], today=date.today(), good_law=flags, top_k=3)
    answer = FakeSynthesizer().synthesise(
        query=Query(
            raw="Is Plessy v. Ferguson still good law?", jurisdiction="US", as_of=date.today()
        ),
        candidates=ranked,
        good_law=flags,
    )
    assert answer.caveats, "answer should surface at least one caveat"
    joined = " ".join(answer.caveats).lower()
    assert "overruled" in joined
    assert "brown" in joined


def test_gate_resumable_no_double_count(landmark_graph) -> None:
    """Re-running the bulk pass picks up zero edges (all labelled)."""
    store, _, _ = landmark_graph
    _run_rules_pass(store)
    second = list(iter_neutral_cites(only_with_sentence=True, store=store))
    assert second == []
