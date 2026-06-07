"""Tier 1 rule classifier — high-precision cue phrases."""

from __future__ import annotations

import pytest

from crimellm.clg.link.treatment_base import EdgeContext
from crimellm.clg.link.treatment_rules import RuleTreatmentClassifier


def _edge(sentence: str) -> EdgeContext:
    return EdgeContext(
        citing_case_id="cl-citing",
        cited_case_id="cl-cited",
        citing_sentence=sentence,
    )


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Plessy v. Ferguson is overruled to the extent it conflicts.", "overruled"),
        ("We overrule the earlier holding.", "overruled"),
        ("That rationale was abrogated by the 2018 amendment.", "overruled"),
        ("The trial court's order is reversed.", "reversed"),
        ("We reverse the dismissal.", "reversed"),
        ("These cases cast doubt on the continuing validity of the test.", "doubted"),
        ("We decline to follow the reasoning of the prior panel.", "not_followed"),
        ("The defendant's conduct is distinguishable from the conduct in Doe.", "distinguished"),
        ("We distinguish the present facts.", "distinguished"),
        ("The judgment is affirmed.", "affirmed"),
        ("We affirm the lower court.", "affirmed"),
        ("Applying the test in Chevron, we hold ...", "applied"),
        ("We follow the holding of the en banc court.", "followed"),
        ("Following the holding in Smith, we conclude ...", "followed"),
        ("See Smith v. Jones, 123 F.3d 456.", "considered"),
    ],
)
def test_rule_classifier_picks_label(sentence: str, expected: str) -> None:
    res = RuleTreatmentClassifier().classify(_edge(sentence))
    assert res is not None, f"rules should fire for: {sentence!r}"
    assert res.label == expected
    assert res.source == "rules"
    assert 0.5 <= res.confidence <= 1.0


def test_rule_classifier_abstains_on_neutral_text() -> None:
    res = RuleTreatmentClassifier().classify(_edge("The plaintiff also alleged a contract claim."))
    assert res is None


def test_rule_classifier_abstains_on_empty_sentence() -> None:
    res = RuleTreatmentClassifier().classify(_edge(""))
    assert res is None


def test_rule_classifier_batch_returns_one_per_input() -> None:
    rc = RuleTreatmentClassifier()
    out = rc.classify_batch(
        [
            _edge("We overrule the prior decision."),
            _edge(""),
            _edge("The dismissal is affirmed."),
        ]
    )
    assert len(out) == 3
    assert out[0] and out[0].label == "overruled"
    assert out[1] is None
    assert out[2] and out[2].label == "affirmed"
