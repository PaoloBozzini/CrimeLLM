"""Phase 10: gold-set filters + DK/EU/cross-jurisdiction question coverage.

Verifies:
* Phase 10 seed.yaml carries balanced coverage across US / UK / EU / DK
  plus cross-jurisdiction (XJ) questions.
* ``GoldSet.filter_by_jurisdiction`` + ``filter_by_task_type`` behave
  correctly (single jurisdiction, multi, ALL/*, XJ semantics).
* CLI ``clg eval --jurisdiction DK,EU`` + ``--task-type good_law`` slice
  the set as expected.
* Q-id conventions (DK lbk paths, EU CELEX paths, ECLI) line up with the
  fixture identifier shapes from Phases 3-5.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from crimellm.clg.cli import app
from crimellm.clg.eval.schema import GoldSet, load_gold_set

SEED = Path("data/eval/seed.yaml")

runner = CliRunner()


# --- gold-set composition -------------------------------------------------


def test_seed_covers_all_jurisdictions():
    gold = load_gold_set(SEED)
    juris = set(gold.jurisdictions())
    # Five active jurisdictions + cross-jurisdiction bucket.
    assert juris == {"US", "UK", "EU", "DK", "XJ"}


def test_seed_meets_phase10_minimums():
    gold = load_gold_set(SEED)
    by_j = {j: gold.filter_by_jurisdiction([j]) for j in ("US", "UK", "EU", "DK")}
    xj = gold.filter_by_jurisdiction(["XJ"])
    # Phase 10 spec: ≥10 DA, ≥5 EU, ≥3 cross-jurisdiction.
    assert len(by_j["DK"]) >= 10, f"need ≥10 DK questions, got {len(by_j['DK'])}"
    assert len(by_j["EU"]) >= 5, f"need ≥5 EU questions, got {len(by_j['EU'])}"
    assert len(xj) >= 3, f"need ≥3 cross-jurisdiction questions, got {len(xj)}"
    # Existing US/UK gate kept.
    assert len(by_j["US"]) >= 2
    assert len(by_j["UK"]) >= 4


def test_seed_all_task_types_present():
    gold = load_gold_set(SEED)
    types = {q.task_type for q in gold.questions}
    assert types == {
        "single_fact",
        "multi_hop",
        "as_of_date",
        "good_law",
        "no_fabrication",
    }


def test_dk_questions_use_correct_id_shapes():
    """DK Provision ids must follow ``dk/<doc_type>/<year>/<num>/section/§<N>...``
    from Phase 4. Case ids are ECLI:DK or Ufr-normalised."""
    gold = load_gold_set(SEED)
    dk = gold.filter_by_jurisdiction(["DK"])
    for q in dk:
        for auth in q.expected_authorities:
            assert (
                auth.startswith("dk/")
                or auth.startswith("ECLI:DK:")
                or auth.startswith("U.")
            ), f"{q.id}: unexpected DK authority shape {auth!r}"


def test_eu_questions_use_correct_id_shapes():
    gold = load_gold_set(SEED)
    eu = gold.filter_by_jurisdiction(["EU"])
    for q in eu:
        for auth in q.expected_authorities:
            assert (
                auth.startswith("eu/celex/")
                or auth.startswith("eu/treaty/")
                or auth.startswith("ECLI:EU:")
            ), f"{q.id}: unexpected EU authority shape {auth!r}"


def test_cross_jurisdiction_questions_span_dk_and_eu():
    gold = load_gold_set(SEED)
    xj = gold.filter_by_jurisdiction(["XJ"])
    for q in xj:
        has_dk = any(
            a.startswith("dk/")
            or a.startswith("ECLI:DK:")
            or a.startswith("U.")
            for a in q.expected_authorities
        )
        has_eu = any(
            a.startswith("eu/")
            or a.startswith("ECLI:EU:")
            for a in q.expected_authorities
        )
        assert has_dk and has_eu, (
            f"{q.id}: cross-jurisdiction question should reference both "
            f"DK and EU authorities (got {q.expected_authorities})"
        )


def test_dk_good_law_uses_civil_law_labels():
    """Civil-law DK questions must use ``departed_from`` / ``criticised``,
    never ``overruled`` / ``reversed``."""
    gold = load_gold_set(SEED)
    dk = gold.filter_by_jurisdiction(["DK"])
    for q in dk:
        for label in q.expected_good_law.values():
            assert label not in {"overruled", "reversed"}, (
                f"{q.id}: civil-law DK should not use common-law label {label!r}"
            )


# --- filter helpers ------------------------------------------------------


def test_filter_by_single_jurisdiction():
    gold = load_gold_set(SEED)
    dk = gold.filter_by_jurisdiction(["DK"])
    assert all(q.jurisdiction == "DK" for q in dk)
    assert len(dk) >= 10


def test_filter_by_multi_jurisdiction():
    gold = load_gold_set(SEED)
    dk_eu = gold.filter_by_jurisdiction(["DK", "EU"])
    juris = {q.jurisdiction for q in dk_eu}
    assert juris == {"DK", "EU"}


def test_filter_includes_xj_when_requested():
    gold = load_gold_set(SEED)
    dk_xj = gold.filter_by_jurisdiction(["DK", "XJ"])
    has_xj = any(q.jurisdiction is None for q in dk_xj)
    has_dk = any(q.jurisdiction == "DK" for q in dk_xj)
    assert has_xj and has_dk


def test_filter_excludes_xj_by_default():
    gold = load_gold_set(SEED)
    dk_only = gold.filter_by_jurisdiction(["DK"])
    assert all(q.jurisdiction == "DK" for q in dk_only)


def test_filter_all_token_keeps_everything():
    gold = load_gold_set(SEED)
    after = gold.filter_by_jurisdiction(["ALL"])
    assert len(after) == len(gold)


def test_filter_star_token_keeps_everything():
    gold = load_gold_set(SEED)
    after = gold.filter_by_jurisdiction(["*"])
    assert len(after) == len(gold)


def test_filter_case_insensitive():
    gold = load_gold_set(SEED)
    upper = gold.filter_by_jurisdiction(["DK"])
    lower = gold.filter_by_jurisdiction(["dk"])
    assert len(upper) == len(lower) > 0


def test_filter_by_task_type():
    gold = load_gold_set(SEED)
    good_law = gold.filter_by_task_type(["good_law"])
    assert all(q.task_type == "good_law" for q in good_law)
    assert len(good_law) >= 3  # Plessy x2, Keck, Højesteret fraveget


def test_filter_returns_new_goldset():
    gold = load_gold_set(SEED)
    filtered = gold.filter_by_jurisdiction(["DK"])
    assert filtered is not gold
    assert filtered.name == gold.name
    assert filtered.version == gold.version


def test_chain_filters():
    gold = load_gold_set(SEED)
    dk_good_law = gold.filter_by_jurisdiction(["DK"]).filter_by_task_type(["good_law"])
    for q in dk_good_law:
        assert q.jurisdiction == "DK"
        assert q.task_type == "good_law"
    assert len(dk_good_law) >= 2


def test_empty_filter_returns_empty_set():
    gold = load_gold_set(SEED)
    empty = gold.filter_by_jurisdiction(["ZZ"])
    assert len(empty) == 0


# --- CLI smoke -----------------------------------------------------------


def test_eval_cli_help_shows_jurisdiction_flag():
    r = runner.invoke(app, ["eval", "--help"])
    assert r.exit_code == 0
    assert "--jurisdiction" in r.stdout
    assert "--task-type" in r.stdout


def test_eval_cli_rejects_empty_filter(tmp_path):
    """When the filter produces zero questions, the CLI should refuse
    instead of silently emitting a 0-question report."""
    r = runner.invoke(
        app,
        [
            "eval",
            "--gold-set",
            str(SEED),
            "--jurisdiction",
            "ZZ",
            "--synth",
            "fake",
        ],
    )
    assert r.exit_code != 0
    combined = r.stderr + r.output
    assert "no questions matched" in combined


# --- regression: existing US/UK questions untouched ----------------------


def test_us_uk_legacy_questions_present():
    """Phase 10 expansion must NOT have dropped the original US / UK
    gate questions."""
    gold = load_gold_set(SEED)
    ids = {q.id for q in gold.questions}
    for legacy in (
        "us-good-law-plessy",
        "us-good-law-named-overruler",
        "uk-single-fact-fraud-penalty-current",
        "uk-as-of-fraud-penalty-2010",
        "uk-multi-hop-r-v-smith-2015",
        "uk-no-fabrication-irrelevant",
    ):
        assert legacy in ids, f"legacy gold question `{legacy}` removed"


# --- gold set integrity (across all questions) ---------------------------


@pytest.mark.parametrize(
    "gold_set",
    [load_gold_set(SEED)],
    ids=["seed"],
)
def test_gold_question_ids_unique(gold_set):
    ids = [q.id for q in gold_set.questions]
    assert len(ids) == len(set(ids)), "duplicate gold question id detected"
