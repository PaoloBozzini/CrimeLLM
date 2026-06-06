"""Cascade orchestrator — escalation logic + telemetry."""

from __future__ import annotations

from collections.abc import Sequence

from crimellm.clg.link.treatment_base import (
    EdgeContext,
    TreatmentClassifier,
    TreatmentResult,
)
from crimellm.clg.link.treatment_cascade import CascadeClassifier


def _edge(idx: int) -> EdgeContext:
    return EdgeContext(
        citing_case_id=f"cl-citing-{idx}",
        cited_case_id=f"cl-cited-{idx}",
        citing_sentence=f"sentence {idx}",
    )


class _FixedClassifier(TreatmentClassifier):
    """Returns a scripted result per edge — handy for cascade tests."""

    def __init__(self, name: str, scripted: Sequence[TreatmentResult | None]):
        self.name = name
        self._scripted = list(scripted)
        self.calls = 0

    def classify_batch(self, edges):  # noqa: ANN001
        self.calls += 1
        out = []
        for i, _ in enumerate(edges):
            out.append(self._scripted[i] if i < len(self._scripted) else None)
        return out


def _res(name: str, label: str, conf: float) -> TreatmentResult:
    return TreatmentResult(label=label, confidence=conf, source=name)  # type: ignore[arg-type]


def test_cascade_short_circuits_on_high_confidence() -> None:
    tier1 = _FixedClassifier(
        "rules", [_res("rules", "overruled", 0.97), _res("rules", "affirmed", 0.95)]
    )
    tier2 = _FixedClassifier("llm", [_res("llm", "neutral", 0.5)] * 2)
    cascade = CascadeClassifier([(tier1, 0.9), (tier2, 0.7)])
    report = cascade.classify([_edge(0), _edge(1)])

    assert len(report.results) == 2
    assert [r.label for r in report.results] == ["overruled", "affirmed"]
    # tier2 should never be hit because tier1 was confident on both.
    assert tier2.calls == 0
    assert {t.accepted_tier for t in report.telemetry} == {"rules"}


def test_cascade_escalates_on_low_confidence_and_abstention() -> None:
    tier1 = _FixedClassifier(
        "rules",
        [
            _res("rules", "applied", 0.55),  # below threshold -> escalate
            None,  # abstain -> escalate
        ],
    )
    tier2 = _FixedClassifier(
        "llm",
        [
            _res("llm", "applied", 0.85),  # accepts
            _res("llm", "neutral", 0.9),  # accepts
        ],
    )
    cascade = CascadeClassifier([(tier1, 0.9), (tier2, 0.7)])
    report = cascade.classify([_edge(0), _edge(1)])

    assert tier1.calls == 1
    assert tier2.calls == 1
    assert [r.label for r in report.results] == ["applied", "neutral"]
    assert [r.source for r in report.results] == ["llm", "llm"]


def test_cascade_keeps_best_fallback_when_every_tier_fails() -> None:
    tier1 = _FixedClassifier("rules", [_res("rules", "applied", 0.4)])
    tier2 = _FixedClassifier("llm", [_res("llm", "applied", 0.6)])
    cascade = CascadeClassifier([(tier1, 0.9), (tier2, 0.9)])
    report = cascade.classify([_edge(0)])

    # Neither tier hit the threshold; best fallback (tier2, conf=0.6) wins.
    assert report.results[0].label == "applied"
    assert report.results[0].confidence == 0.6
    assert report.telemetry[0].fallback_used is True


def test_cascade_budget_caps_expensive_tier() -> None:
    expensive = _FixedClassifier(
        "anthropic",
        [
            _res("anthropic", "overruled", 0.99),
            _res("anthropic", "overruled", 0.99),
            _res("anthropic", "overruled", 0.99),
        ],
    )
    cheap = _FixedClassifier(
        "rules",
        [None, None, None],
    )
    cascade = CascadeClassifier(
        [(cheap, 0.9), (expensive, 0.9)],
        budget_per_tier={"anthropic": 1},
    )
    report = cascade.classify([_edge(0), _edge(1), _edge(2)])

    # Only the first edge got the expensive tier; the rest fall back to neutral.
    accepted_tiers = [t.accepted_tier for t in report.telemetry]
    assert accepted_tiers.count("anthropic") == 1
    assert "cascade:no-tier" in {r.source for r in report.results}


def test_cascade_by_tier_count_summary() -> None:
    tier1 = _FixedClassifier("rules", [_res("rules", "overruled", 0.97)] * 5)
    cascade = CascadeClassifier([(tier1, 0.9)])
    report = cascade.classify([_edge(i) for i in range(5)])
    assert report.by_tier() == {"rules": 5}
