"""Gold-set loader + dataclass shape."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.eval.schema import GoldQuestion, GoldSet, load_gold_set

SEED = Path(__file__).parents[2] / "data" / "eval" / "seed.yaml"


def test_seed_gold_set_loads() -> None:
    gold = load_gold_set(SEED)
    assert isinstance(gold, GoldSet)
    assert gold.name == "clg-seed"
    assert len(gold) >= 6
    by_task = {q.task_type for q in gold}
    # All four brief-mandated task types are covered, plus the fabrication probe.
    assert {"single_fact", "as_of_date", "multi_hop", "good_law", "no_fabrication"}.issubset(
        by_task
    )


def test_seed_questions_have_required_fields() -> None:
    gold = load_gold_set(SEED)
    for q in gold:
        assert q.id
        assert q.question
        assert q.task_type in {
            "single_fact",
            "multi_hop",
            "as_of_date",
            "good_law",
            "no_fabrication",
        }
        # as_of date must parse to a real date when set.
        if q.as_of is not None:
            assert isinstance(q.as_of, date)


def test_load_gold_set_rejects_top_level_list(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- id: q1\n  question: x\n  task_type: single_fact\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level YAML must be a mapping"):
        load_gold_set(bad)


def test_load_gold_set_rejects_missing_required_field(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: t\nquestions:\n  - id: q1\n    question: x\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field"):
        load_gold_set(bad)


def test_load_gold_set_parses_iso_date(tmp_path) -> None:
    f = tmp_path / "g.yaml"
    f.write_text(
        "name: t\nquestions:\n  - id: q1\n    question: x\n    task_type: single_fact\n"
        "    as_of: 2020-01-01\n",
        encoding="utf-8",
    )
    g = load_gold_set(f)
    assert g.questions[0].as_of == date(2020, 1, 1)


def test_question_defaults() -> None:
    q = GoldQuestion(id="q", question="?", task_type="single_fact")
    assert q.expected_authorities == []
    assert q.expected_good_law == {}
    assert q.expected_treating_case is None
    assert q.tags == []
