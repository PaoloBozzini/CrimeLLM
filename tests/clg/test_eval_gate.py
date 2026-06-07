"""Phase 6 gate — run the harness end-to-end, hit the four metric families.

Loads a focused subset of the seed gold set against the same Phase 2+5
fixtures the earlier gates use, runs ``run_eval`` with FakeEmbedder +
FakeSynthesizer, and asserts the report carries values for every metric the
brief calls out.

Auto-skipped when Neo4j is unreachable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.embed.chunker import chunk_provision
from crimellm.clg.embed.embedder import FakeEmbedder
from crimellm.clg.eval import GoldQuestion, GoldSet, run_eval
from crimellm.clg.eval.report import to_json, to_markdown
from crimellm.clg.graph.loaders import (
    iter_neutral_cites,
    load_cases,
    load_chunks,
    load_citations,
    load_instruments,
    load_interprets,
    load_provisions,
    vector_index_dim,
    write_treatments,
)
from crimellm.clg.graph.schema import apply_schema, rebuild_vector_index
from crimellm.clg.link import CascadeClassifier, EdgeContext, RuleTreatmentClassifier
from crimellm.clg.models import Case, Citation, Provenance
from crimellm.clg.parse import find_case_law as FCL
from crimellm.clg.parse import legislation_uk as LEG
from crimellm.clg.retrieval import FakeSynthesizer

LEG_FIX = Path(__file__).parent / "fixtures" / "legislation_uk"
FCL_FIX = Path(__file__).parent / "fixtures" / "find_case_law"


def _run_treatment_pass(store) -> None:
    """Run rules-only cascade so Phase 6's good-law check has data."""
    cascade = CascadeClassifier([(RuleTreatmentClassifier(), 0.85)])
    rows = list(iter_neutral_cites(only_with_sentence=True, store=store))
    if not rows:
        return
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


@pytest.fixture
def gate_world(neo4j_store):
    apply_schema(neo4j_store)
    if vector_index_dim(neo4j_store) != 1024:
        rebuild_vector_index(1024, drop_chunks=True, store=neo4j_store)
    with neo4j_store.session() as s:
        s.run("MATCH (n) WHERE n:Instrument OR n:Provision OR n:Case OR n:Chunk DETACH DELETE n")

    # UK legislation — Fraud Act (2 versions) + a 2015 judgment that
    # interprets s.2 and s.3.
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

    case, refs = FCL.parse_judgment_file(FCL_FIX / "ewca-crim-fraud-2015.xml")
    load_cases([case], store=neo4j_store)
    load_interprets([(case.id, case.decision_date, r) for r in refs], store=neo4j_store)

    # Plessy v Ferguson + Brown v Board with explicit overruling citing sentence.
    plessy = Case(
        id="cl-cluster-plessy",
        jurisdiction="US",
        court_id="scotus",
        name="Plessy v. Ferguson",
        decision_date=date(1896, 5, 18),
        citations=["163 U.S. 537"],
        provenance=[
            Provenance(
                source="fixture", source_url="", retrieved_at=date.today(), source_id="plessy"
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
                source="fixture", source_url="", retrieved_at=date.today(), source_id="brown"
            )
        ],
    )
    load_cases([plessy, brown], store=neo4j_store)
    load_citations(
        [
            Citation(
                citing_case_id=brown.id,
                cited_case_id=plessy.id,
                treatment="neutral",
                citing_sentence=(
                    "Plessy v. Ferguson is overruled to the extent it conflicts with this opinion."
                ),
                weight=1.0,
            )
        ],
        store=neo4j_store,
    )
    _run_treatment_pass(neo4j_store)
    return neo4j_store, embedder


def _focused_gold() -> GoldSet:
    """A 4-question subset that exercises one question per task type.

    Uses very short, FakeEmbedder-friendly queries: with hash-based vectors,
    the only way to hit a passage is to query its exact indexed text. The
    statute questions therefore embed the section's chunk text directly.
    """
    fraud_s3_text = (
        "s.3 Penalty A person guilty of fraud is liable on indictment to "
        "imprisonment for a term not exceeding 10 years."
    )
    fraud_s3_enacted_text = (
        "s.3 Penalty A person guilty of fraud is liable on indictment to "
        "imprisonment for a term not exceeding 7 years."
    )
    return GoldSet(
        name="gate-subset",
        description="Phase 6 gate subset",
        questions=[
            GoldQuestion(
                id="g-single",
                question=fraud_s3_text,
                task_type="single_fact",
                jurisdiction="UK",
                as_of=date(2024, 6, 1),
                expected_authorities=["uk/ukpga/2006/35/section/3@current"],
            ),
            GoldQuestion(
                id="g-as-of",
                question=fraud_s3_enacted_text,
                task_type="as_of_date",
                jurisdiction="UK",
                as_of=date(2010, 6, 1),
                expected_authorities=["uk/ukpga/2006/35/section/3@enacted"],
            ),
            GoldQuestion(
                id="g-no-fab",
                question="Quantum computing intellectual property under the Fraud Act 2006.",
                task_type="no_fabrication",
                jurisdiction="UK",
                as_of=date(2024, 6, 1),
                expected_authorities=[],
            ),
        ],
    )


def test_run_eval_returns_full_report(gate_world) -> None:
    store, embedder = gate_world
    report = run_eval(
        _focused_gold(),
        embedder=embedder,
        synthesizer=FakeSynthesizer(),
        store=store,
    )
    assert report.n_questions() == 3
    by_id = {s.question_id: s for s in report.scores}
    # Single-fact: recall must be perfect (the seeded chunk is in the index).
    assert by_id["g-single"].recall_at_k == 1.0
    # As-of-date: the enacted version must be in the surfaced set.
    assert by_id["g-as-of"].as_of_correct is True
    # No-fabrication: even on an out-of-scope query, FakeSynthesizer only
    # quotes ids from the candidate list — it never invents.
    assert by_id["g-no-fab"].fabricated_citations == []


def test_aggregate_covers_brief_metrics(gate_world) -> None:
    """Brief Phase 6 — four metric families surfaced in the aggregate."""
    store, embedder = gate_world
    report = run_eval(
        _focused_gold(),
        embedder=embedder,
        synthesizer=FakeSynthesizer(),
        store=store,
    )
    agg = report.aggregate
    assert agg.recall_at_k_mean is not None
    assert agg.citation_accuracy_mean is not None
    assert agg.as_of_correct_rate is not None
    # fabrication rate is a float (>=0); zero or low is good.
    assert agg.fabrication_rate >= 0.0


def test_to_markdown_renders_table(gate_world) -> None:
    store, embedder = gate_world
    report = run_eval(
        _focused_gold(),
        embedder=embedder,
        synthesizer=FakeSynthesizer(),
        store=store,
    )
    md = to_markdown(report)
    assert "Aggregate" in md
    assert "Per-question" in md
    assert "g-single" in md


def test_to_json_round_trips(gate_world) -> None:
    import json

    store, embedder = gate_world
    report = run_eval(
        _focused_gold(),
        embedder=embedder,
        synthesizer=FakeSynthesizer(),
        store=store,
    )
    payload = json.loads(to_json(report))
    assert payload["gold_set"]["n_questions"] == 3
    assert "aggregate" in payload
    assert "scores" in payload and len(payload["scores"]) == 3
