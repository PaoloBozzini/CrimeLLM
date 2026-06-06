"""Rerank + dedupe + recency curve — pure logic, no Neo4j."""

from __future__ import annotations

from datetime import date

from crimellm.clg.retrieval.rerank import (
    RerankWeights,
    _recency_score,
    dedupe_candidates,
    rerank,
)
from crimellm.clg.retrieval.seed import Candidate


def _cand(**kw) -> Candidate:
    base = dict(
        chunk_id=None,
        text="",
        parent_type="Case",
        parent_id="cl-cluster-1",
        parent_name="Test Case",
        parent_jurisdiction="US",
    )
    base.update(kw)
    return Candidate(**base)  # type: ignore[arg-type]


def test_recency_score_today_is_one() -> None:
    today = date(2024, 6, 1)
    assert abs(_recency_score(today, today=today) - 1.0) < 1e-6


def test_recency_score_decays_over_time() -> None:
    today = date(2024, 6, 1)
    a = _recency_score(date(2020, 6, 1), today=today)  # 4 years
    b = _recency_score(date(2000, 6, 1), today=today)  # 24 years
    assert 0.0 < b < a < 1.0


def test_recency_score_unknown_date_is_zero() -> None:
    assert _recency_score(None, today=date(2024, 6, 1)) == 0.0
    assert _recency_score("not-a-date", today=date(2024, 6, 1)) == 0.0


def test_dedupe_unifies_sources_and_keeps_max_score() -> None:
    a = _cand(parent_id="cl-1", source="seed", base_score=0.4)
    b = _cand(parent_id="cl-1", source="cited_by", base_score=0.7)
    pooled = dedupe_candidates([a, b])
    assert len(pooled) == 1
    out = pooled[0]
    assert out.base_score == 0.7
    assert set(out.extras["sources"]) == {"seed", "cited_by"}


def test_rerank_prefers_higher_vector_score() -> None:
    today = date(2024, 6, 1)
    items = [
        _cand(parent_id="cl-low", base_score=0.2),
        _cand(parent_id="cl-high", base_score=0.9),
    ]
    out = rerank(items, weights=RerankWeights(), today=today)
    assert out[0].parent_id == "cl-high"


def test_rerank_penalises_good_law_flagged_cases() -> None:
    today = date(2024, 6, 1)
    a = _cand(parent_id="cl-flagged", base_score=0.8)
    b = _cand(parent_id="cl-clean", base_score=0.7)
    from crimellm.clg.retrieval.good_law import GoodLawFlag

    flags = {
        "cl-flagged": [
            GoodLawFlag(
                case_id="cl-flagged",
                treatment="overruled",
                treating_case_id="cl-overruler",
                treating_case_name="Overruler",
            )
        ]
    }
    out = rerank([a, b], today=today, good_law=flags)
    # 0.8 - 0.15 = 0.65, vs 0.7 for cl-clean
    assert out[0].parent_id == "cl-clean"
    assert out[1].parent_id == "cl-flagged"


def test_rerank_returns_only_top_k() -> None:
    items = [_cand(parent_id=f"cl-{i}", base_score=i / 10) for i in range(10)]
    out = rerank(items, top_k=3)
    assert len(out) == 3
