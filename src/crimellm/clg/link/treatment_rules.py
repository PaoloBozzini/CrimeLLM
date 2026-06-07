"""Tier 1 — rule classifier. Pattern matches on the citing sentence.

High precision, low recall. Designed to label the easy ~30-40% of edges at
near-zero cost and abstain (return ``None``) on the rest so the cascade
escalates to a smarter tier.

**Per-jurisdiction registry.** Common-law (US / EW / UK / EU) and
civil-law (DK) cite differently; their cue phrases overlap only thinly.
``RULES_BY_JURISDICTION`` is the source of truth — DK and EU register
their rule lists from sibling files (``treatment_rules_dk`` /
``treatment_rules_eu``) on import.

``RuleTreatmentClassifier(jurisdiction="DK")`` pulls the matching list
and **abstains** on edges from other jurisdictions, so a cascade can hold
one rule classifier per enabled jurisdiction and they don't fight each
other. ``RuleTreatmentClassifier(jurisdiction=None)`` keeps the old
behaviour for callers that haven't been updated yet (matches everything
against the common-law rule set; default for backward compat).

Patterns within each rule list are ordered: the first matching pattern
wins. Ordering goes adverse-first so a sentence saying "we follow X, but
distinguish Y" is handled by the most specific cue.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from .treatment_base import (
    EdgeContext,
    TreatmentClassifier,
    TreatmentLabel,
    TreatmentResult,
)


@dataclass(slots=True)
class _Rule:
    label: TreatmentLabel
    pattern: re.Pattern[str]
    confidence: float


def _ci(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Conservative cue phrases. Each one is meant to be unambiguous on its own.
# Borderline patterns get lower confidence so the cascade can override.
COMMON_LAW_RULES: list[_Rule] = [
    # Adverse — strong signal
    _Rule("overruled", _ci(r"\b(is|was|are|now)\s+overruled\b"), 0.97),
    _Rule("overruled", _ci(r"\bwe\s+overrule\b"), 0.97),
    _Rule("overruled", _ci(r"\babrogated\s+by\b"), 0.95),
    _Rule("overruled", _ci(r"\bexpressly\s+overrul"), 0.97),
    _Rule("reversed", _ci(r"\b(is|was|are)\s+reversed\b"), 0.95),
    _Rule("reversed", _ci(r"\bwe\s+reverse\b"), 0.93),
    _Rule("doubted", _ci(r"\b(cast\s+doubt|its\s+(?:continuing\s+)?validity)"), 0.9),
    _Rule("doubted", _ci(r"\bquestionable\s+(?:authority|validity)\b"), 0.9),
    _Rule("not_followed", _ci(r"\bdecline(?:s)?\s+to\s+follow\b"), 0.92),
    _Rule("not_followed", _ci(r"\brefuse(?:s)?\s+to\s+follow\b"), 0.92),
    _Rule("distinguished", _ci(r"\bdistinguishable\s+from\b"), 0.92),
    _Rule("distinguished", _ci(r"\bwe\s+distinguish\b"), 0.9),
    _Rule("distinguished", _ci(r"\bdistinguished\s+from\b"), 0.9),
    # Favourable / neutral
    _Rule("affirmed", _ci(r"\b(is|was|are)\s+affirmed\b"), 0.9),
    _Rule("affirmed", _ci(r"\bwe\s+affirm\b"), 0.9),
    _Rule("applied", _ci(r"\bapplying\s+the\s+(?:rule|test|standard)\s+(?:in|of)\b"), 0.9),
    _Rule("followed", _ci(r"\b(we|the\s+court)\s+follow(?:s|ed)?\b"), 0.85),
    _Rule("followed", _ci(r"\bfollowing\s+the\s+holding\b"), 0.88),
    _Rule("considered", _ci(r"\b(see|cf\.|compare)\b"), 0.6),
]

# Back-compat alias — older code/tests import the plain ``RULES`` symbol.
RULES = COMMON_LAW_RULES

# Per-jurisdiction registry. Common-law jurisdictions share the base set;
# DK and EU sibling files (treatment_rules_dk, treatment_rules_eu) extend
# this dict on import. Lookup is case-insensitive via ``rules_for(j)``.
RULES_BY_JURISDICTION: dict[str, list[_Rule]] = {
    "US": COMMON_LAW_RULES,
    "EW": COMMON_LAW_RULES,
    "UK": COMMON_LAW_RULES,
}


def register_rules(jurisdiction: str, rules: list[_Rule]) -> None:
    """Register a per-jurisdiction rule list. Last-write-wins so a test
    can swap rules without monkey-patching.
    """
    RULES_BY_JURISDICTION[jurisdiction.upper()] = rules


def rules_for(jurisdiction: str | None) -> list[_Rule]:
    """Return the rule list for ``jurisdiction``; common-law default on miss."""
    if jurisdiction is None:
        return COMMON_LAW_RULES
    return RULES_BY_JURISDICTION.get(jurisdiction.upper(), COMMON_LAW_RULES)


class RuleTreatmentClassifier(TreatmentClassifier):
    """Tier 1: lexical cues.

    ``jurisdiction=None`` matches every edge against the common-law rules
    (backward-compatible behaviour). ``jurisdiction="DK"`` pulls the DK
    rule list and abstains on edges from other jurisdictions, so a
    cascade with one classifier per jurisdiction won't double-label.
    """

    name = "rules"

    def __init__(
        self,
        rules: Sequence[_Rule] | None = None,
        *,
        jurisdiction: str | None = None,
    ):
        self.jurisdiction = jurisdiction.upper() if jurisdiction else None
        if rules is not None:
            self.rules = list(rules)
        else:
            self.rules = list(rules_for(self.jurisdiction))
        # Sub-name the tier so cascade telemetry distinguishes
        # ``rules:DK`` from ``rules:EU``.
        if self.jurisdiction:
            self.name = f"rules:{self.jurisdiction}"

    def classify_batch(self, edges: Sequence[EdgeContext]) -> list[TreatmentResult | None]:
        out: list[TreatmentResult | None] = []
        for e in edges:
            out.append(self._classify_one(e))
        return out

    def _matches_jurisdiction(self, edge: EdgeContext) -> bool:
        if self.jurisdiction is None:
            return True
        edge_j = (edge.citing_case_jurisdiction or "").upper()
        # When the edge has no jurisdiction tag, fall through — the
        # classifier still tries to match. Avoids over-eager abstention
        # on legacy graph rows.
        return edge_j == "" or edge_j == self.jurisdiction

    def _classify_one(self, edge: EdgeContext) -> TreatmentResult | None:
        if not self._matches_jurisdiction(edge):
            return None
        sentence = edge.citing_sentence
        if not sentence:
            return None
        start = time.perf_counter()
        for rule in self.rules:
            if rule.pattern.search(sentence):
                return TreatmentResult(
                    label=rule.label,
                    confidence=rule.confidence,
                    source=self.name,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    extras={
                        "matched": rule.pattern.pattern,
                        "jurisdiction": self.jurisdiction or "(any)",
                    },
                )
        return None


# Side-effect: pull in DK + EU rule modules so their registry entries
# materialise the moment ``from crimellm.clg.link import ...`` runs.
from . import treatment_rules_dk, treatment_rules_eu  # noqa: F401, E402
