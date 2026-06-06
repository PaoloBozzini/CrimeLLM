"""Tier 1 — rule classifier. Pattern matches on the citing sentence.

High precision, low recall. Designed to label the easy ~30-40% of edges at
near-zero cost and abstain (return ``None``) on the rest so the cascade
escalates to a smarter tier.

Patterns are ordered: the first matching pattern wins. Ordering goes
adverse-first so a sentence saying "we follow X, but distinguish Y" is
handled by the most specific cue.
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
RULES: list[_Rule] = [
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


class RuleTreatmentClassifier(TreatmentClassifier):
    """Tier 1: lexical cues."""

    name = "rules"

    def __init__(self, rules: Sequence[_Rule] = RULES):
        self.rules = list(rules)

    def classify_batch(self, edges: Sequence[EdgeContext]) -> list[TreatmentResult | None]:
        out: list[TreatmentResult | None] = []
        for e in edges:
            out.append(self._classify_one(e))
        return out

    def _classify_one(self, edge: EdgeContext) -> TreatmentResult | None:
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
                    extras={"matched": rule.pattern.pattern},
                )
        return None
