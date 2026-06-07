"""Phase 6: per-jurisdiction treatment rules + cascade routing.

Covers:
* Registry shape: DK + EU register on import; common-law jurisdictions
  share the base list.
* DK classifier picks DK cues; abstains on US edges.
* EU classifier picks both common-law + CJEU-specific cues; abstains on DK.
* Common-law (None jurisdiction) backward-compat unchanged.
* Cascade with one classifier per jurisdiction routes each edge to the
  right tier.
"""

from __future__ import annotations

import pytest

from crimellm.clg.link.treatment_base import EdgeContext
from crimellm.clg.link.treatment_cascade import CascadeClassifier
from crimellm.clg.link.treatment_rules import (
    COMMON_LAW_RULES,
    RULES_BY_JURISDICTION,
    RuleTreatmentClassifier,
    rules_for,
)


# --- registry shape -------------------------------------------------------


def test_registry_has_all_jurisdictions():
    assert "US" in RULES_BY_JURISDICTION
    assert "UK" in RULES_BY_JURISDICTION
    assert "EW" in RULES_BY_JURISDICTION
    assert "DK" in RULES_BY_JURISDICTION
    assert "EU" in RULES_BY_JURISDICTION


def test_common_law_jurisdictions_share_rule_list():
    # US / UK / EW point at the same in-memory list — no duplication.
    assert RULES_BY_JURISDICTION["US"] is COMMON_LAW_RULES
    assert RULES_BY_JURISDICTION["UK"] is COMMON_LAW_RULES
    assert RULES_BY_JURISDICTION["EW"] is COMMON_LAW_RULES


def test_dk_rules_distinct_from_common_law():
    dk = RULES_BY_JURISDICTION["DK"]
    assert dk is not COMMON_LAW_RULES
    # DK shouldn't carry the common-law "overrule" labels — civil-law
    # courts depart from, not overrule.
    assert not any(r.label == "overruled" for r in dk)
    assert not any(r.label == "reversed" for r in dk)
    assert any(r.label == "departed_from" for r in dk)
    assert any(r.label == "criticised" for r in dk)


def test_eu_rules_extend_common_law():
    eu = RULES_BY_JURISDICTION["EU"]
    # EU set is strictly larger than common-law base.
    assert len(eu) > len(COMMON_LAW_RULES)
    # Includes CJEU-specific departure cue.
    assert any("departing" in r.pattern.pattern for r in eu)
    # Inherits common-law overruled/reversed labels too.
    assert any(r.label == "overruled" for r in eu)


def test_rules_for_unknown_falls_back_to_common_law():
    assert rules_for("XX") is COMMON_LAW_RULES
    assert rules_for(None) is COMMON_LAW_RULES


def test_rules_for_case_insensitive():
    assert rules_for("dk") is RULES_BY_JURISDICTION["DK"]
    assert rules_for("eu") is RULES_BY_JURISDICTION["EU"]


# --- DK classifier --------------------------------------------------------


def _edge(sentence: str, jurisdiction: str | None = None) -> EdgeContext:
    return EdgeContext(
        citing_case_id="a",
        cited_case_id="b",
        citing_sentence=sentence,
        citing_case_jurisdiction=jurisdiction,
    )


def test_dk_classifier_picks_dk_followed():
    clf = RuleTreatmentClassifier(jurisdiction="DK")
    edge = _edge(
        "Højesteret afgjorde sagen i overensstemmelse med U.2010.456H.",
        jurisdiction="DK",
    )
    res = clf.classify(edge)
    assert res is not None
    assert res.label == "followed"
    assert res.source == "rules:DK"
    assert res.extras.get("jurisdiction") == "DK"


def test_dk_classifier_picks_departed_from():
    clf = RuleTreatmentClassifier(jurisdiction="DK")
    edge = _edge(
        "Højesteret har fraveget den tidligere praksis i U.1995.123H.",
        jurisdiction="DK",
    )
    res = clf.classify(edge)
    assert res is not None
    assert res.label == "departed_from"


def test_dk_classifier_picks_criticised():
    clf = RuleTreatmentClassifier(jurisdiction="DK")
    edge = _edge(
        "Afgørelsen er kritiseret af landsretten i en senere sag.",
        jurisdiction="DK",
    )
    res = clf.classify(edge)
    assert res is not None
    assert res.label == "criticised"


def test_dk_classifier_abstains_on_us_edge():
    clf = RuleTreatmentClassifier(jurisdiction="DK")
    edge = _edge(
        "Højesteret afgjorde sagen i overensstemmelse med U.2010.456H.",
        jurisdiction="US",  # mis-tagged edge — DK classifier abstains
    )
    assert clf.classify(edge) is None


def test_dk_classifier_matches_when_edge_jurisdiction_absent():
    """Legacy edges without a jurisdiction tag still get matched —
    abstention only triggers on an explicit non-matching tag."""
    clf = RuleTreatmentClassifier(jurisdiction="DK")
    edge = _edge(
        "Afgørelsen er kritiseret af landsretten.",
        jurisdiction=None,
    )
    res = clf.classify(edge)
    assert res is not None
    assert res.label == "criticised"


# --- EU classifier --------------------------------------------------------


def test_eu_classifier_picks_cjeu_departure():
    clf = RuleTreatmentClassifier(jurisdiction="EU")
    edge = _edge(
        "The Court is departing from its earlier ruling in Dassonville.",
        jurisdiction="EU",
    )
    res = clf.classify(edge)
    assert res is not None
    assert res.label == "departed_from"
    assert res.source == "rules:EU"


def test_eu_classifier_picks_settled_case_law():
    clf = RuleTreatmentClassifier(jurisdiction="EU")
    edge = _edge(
        "It is settled case-law that Article 101 TFEU applies broadly.",
        jurisdiction="EU",
    )
    res = clf.classify(edge)
    assert res is not None
    assert res.label == "followed"


def test_eu_classifier_inherits_common_law_overruled():
    clf = RuleTreatmentClassifier(jurisdiction="EU")
    edge = _edge(
        "The earlier ruling is expressly overruled today.",
        jurisdiction="EU",
    )
    res = clf.classify(edge)
    assert res is not None
    assert res.label == "overruled"


def test_eu_classifier_abstains_on_dk_edge():
    clf = RuleTreatmentClassifier(jurisdiction="EU")
    edge = _edge(
        "It is settled case-law that Article 101 TFEU applies.",
        jurisdiction="DK",
    )
    assert clf.classify(edge) is None


# --- common-law (None) backward compat -----------------------------------


def test_jurisdiction_none_matches_anything():
    """Backward-compat: RuleTreatmentClassifier() with no jurisdiction
    keeps the legacy behaviour — matches every edge against common-law."""
    clf = RuleTreatmentClassifier()
    edge_us = _edge("We overrule that holding now.", jurisdiction="US")
    edge_no = _edge("We overrule that holding now.", jurisdiction=None)
    edge_dk = _edge("We overrule that holding now.", jurisdiction="DK")
    for e in (edge_us, edge_no, edge_dk):
        res = clf.classify(e)
        assert res is not None
        assert res.label == "overruled"
        assert res.source == "rules"  # un-suffixed


# --- cascade routing -----------------------------------------------------


def test_cascade_with_one_classifier_per_jurisdiction_routes_correctly():
    """Build a cascade with US + DK + EU rule classifiers. Each edge
    should be labelled by the matching-jurisdiction tier (telemetry
    reports the source name)."""
    cascade = CascadeClassifier(
        [
            (RuleTreatmentClassifier(jurisdiction="US"), 0.85),
            (RuleTreatmentClassifier(jurisdiction="DK"), 0.85),
            (RuleTreatmentClassifier(jurisdiction="EU"), 0.85),
        ]
    )
    edges = [
        _edge("We overrule that case today.", jurisdiction="US"),
        _edge("Højesteret har fraveget U.2010.456H.", jurisdiction="DK"),
        _edge("Settled case-law of the Court.", jurisdiction="EU"),
    ]
    report = cascade.classify(edges)
    assert [r.label for r in report.results] == ["overruled", "departed_from", "followed"]
    assert [t.accepted_tier for t in report.telemetry] == [
        "rules:US",
        "rules:DK",
        "rules:EU",
    ]


def test_cascade_unknown_jurisdiction_falls_through_to_neutral():
    """Edge with an unknown jurisdiction tag and no matching tier — all
    classifiers abstain → cascade falls back to neutral."""
    cascade = CascadeClassifier(
        [(RuleTreatmentClassifier(jurisdiction="DK"), 0.85)]
    )
    edges = [_edge("we follow that holding.", jurisdiction="XX")]
    report = cascade.classify(edges)
    assert report.results[0].label == "neutral"
    assert report.results[0].source == "cascade:no-tier"


@pytest.mark.parametrize("j", ["DK", "EU", "US", "UK", "EW"])
def test_classifier_constructible_for_every_enabled_jurisdiction(j):
    clf = RuleTreatmentClassifier(jurisdiction=j)
    assert clf.jurisdiction == j
    assert clf.name == f"rules:{j}"
    assert clf.rules  # non-empty
