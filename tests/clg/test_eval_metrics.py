"""Per-question metrics — no Neo4j, no retrieval, just deterministic math."""

from __future__ import annotations

from crimellm.clg.eval.metrics import (
    aggregate,
    as_of_correct,
    citation_accuracy,
    good_law_precision_recall,
    recall_at_k,
    score_question,
)
from crimellm.clg.eval.schema import GoldQuestion
from crimellm.clg.retrieval.seed import Candidate
from crimellm.clg.retrieval.synthesize import Answer


def _candidate(parent_id: str, parent_type: str = "Case", name: str | None = None) -> Candidate:
    return Candidate(
        chunk_id=None,
        text="",
        parent_type=parent_type,
        parent_id=parent_id,
        parent_name=name or parent_id,
        parent_jurisdiction="US",
    )


def _answer(
    text: str,
    used: list[Candidate],
    citations: list[str] | None = None,
    caveats: list[str] | None = None,
) -> Answer:
    return Answer(
        question="?",
        text=text,
        citations=list(citations or []),
        caveats=list(caveats or []),
        used_candidates=list(used),
        model="fake",
    )


def test_recall_at_k_counts_surfaced_authorities() -> None:
    ans = _answer("x", [_candidate("cl-1"), _candidate("cl-2")], citations=["cl-1"])
    assert recall_at_k(ans, ["cl-1", "cl-2"]) == 1.0
    assert recall_at_k(ans, ["cl-3"]) == 0.0
    assert recall_at_k(ans, ["cl-1", "cl-3"]) == 0.5


def test_recall_at_k_returns_none_when_no_expected() -> None:
    ans = _answer("x", [_candidate("cl-1")])
    assert recall_at_k(ans, []) is None


def test_citation_accuracy_perfect_when_only_valid_cites() -> None:
    ans = _answer("Top [cl-1]", [_candidate("cl-1")], citations=["cl-1"])
    assert citation_accuracy(ans) == 1.0


def test_citation_accuracy_drops_on_fabrication() -> None:
    ans = _answer(
        "Top [cl-1] [made-up]",
        [_candidate("cl-1")],
        citations=["cl-1"],
        caveats=["WARNING — model emitted citations not present in retrieved context: made-up"],
    )
    assert citation_accuracy(ans) == 0.5


def test_good_law_precision_recall_clean_hit() -> None:
    q = GoldQuestion(
        id="q",
        question="?",
        task_type="good_law",
        expected_good_law={"cl-plessy": "overruled"},
        expected_treating_case="cl-brown",
    )
    ans = _answer(
        "Plessy was overruled by Brown [cl-brown].",
        [_candidate("cl-plessy")],
        citations=["cl-plessy", "cl-brown"],
        caveats=["Plessy v. Ferguson [cl-plessy] — overruled by Brown v. Board [cl-brown]."],
    )
    p, r, named = good_law_precision_recall(ans, q)
    assert r == 1.0
    assert p == 1.0
    assert named is True


def test_good_law_named_treater_missing_when_not_mentioned() -> None:
    q = GoldQuestion(
        id="q",
        question="?",
        task_type="good_law",
        expected_good_law={"cl-plessy": "overruled"},
        expected_treating_case="cl-brown",
    )
    ans = _answer(
        "Plessy was overruled by a later case.",
        [_candidate("cl-plessy")],
        citations=["cl-plessy"],
        caveats=["Plessy v. Ferguson [cl-plessy] — overruled by unknown."],
    )
    _, _, named = good_law_precision_recall(ans, q)
    assert named is False


def test_good_law_returns_none_when_not_applicable() -> None:
    q = GoldQuestion(id="q", question="?", task_type="single_fact")
    ans = _answer("x", [_candidate("cl-1")])
    p, r, named = good_law_precision_recall(ans, q)
    assert (p, r, named) == (None, None, None)


def test_as_of_correct_requires_all_expected() -> None:
    ans = _answer(
        "x",
        [
            _candidate("uk/...@enacted", parent_type="Provision"),
            _candidate("uk/...@current", parent_type="Provision"),
        ],
    )
    assert as_of_correct(ans, ["uk/...@enacted"]) is True
    assert as_of_correct(ans, ["uk/...@missing"]) is False
    assert as_of_correct(ans, []) is None


def test_score_question_no_fabrication_inverts_contract() -> None:
    q = GoldQuestion(
        id="adv",
        question="invent something",
        task_type="no_fabrication",
        expected_authorities=[],
    )
    ans = _answer(
        "made-up [pretend-id]",
        used=[_candidate("cl-real")],
        citations=[],
        caveats=["WARNING — model emitted citations not present in retrieved context: pretend-id"],
    )
    s = score_question(q, ans)
    assert s.fabricated_citations == ["pretend-id"]
    assert s.citation_accuracy == 0.0
    assert s.recall_at_k is None  # nothing was expected


def test_aggregate_handles_empty_and_partial() -> None:
    assert aggregate([]).n == 0

    q1 = GoldQuestion(id="q1", question="?", task_type="single_fact", expected_authorities=["cl-1"])
    a1 = _answer("Top [cl-1]", [_candidate("cl-1")], citations=["cl-1"])
    s1 = score_question(q1, a1)

    q2 = GoldQuestion(id="q2", question="?", task_type="no_fabrication")
    a2 = _answer(
        "bad [made-up]",
        [_candidate("cl-real")],
        citations=[],
        caveats=["WARNING — model emitted citations not present in retrieved context: made-up"],
    )
    s2 = score_question(q2, a2)

    agg = aggregate([s1, s2])
    assert agg.n == 2
    assert agg.recall_at_k_mean == 1.0  # only one applies
    assert agg.fabrication_rate == 0.5
