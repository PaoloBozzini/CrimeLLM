"""Tier 1 — CJEU cue phrases (English judgment text).

CJEU writes in many languages but English is the cross-border lingua
franca and what most legal-tech NLP pipelines target first. The Court's
formulaic style makes the cue phrases unusually stable:

* ``settled case-law`` / ``it is settled case-law`` — followed
* ``departing from`` — departed_from (cf. *Keck Mithouard* C-267/91
  departing from *Dassonville* 8/74)
* ``restated in`` / ``as the Court held in`` — applied
* ``the Court has consistently held`` — followed

We extend (not replace) the common-law rule set because CJEU adopts most
of the same verbs (``distinguished``, ``affirmed``, etc.). Multilingual
support for French / German judgment bodies is Phase 6.5 work.
"""

from __future__ import annotations

import re

from .treatment_base import TreatmentLabel
from .treatment_rules import COMMON_LAW_RULES, _Rule, register_rules


def _ci(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _r(label: TreatmentLabel, pattern: str, confidence: float) -> _Rule:
    return _Rule(label=label, pattern=_ci(pattern), confidence=confidence)


EU_EXTRA_RULES: list[_Rule] = [
    # Departure — CJEU does explicitly depart from its own precedent.
    _r("departed_from", r"\bdeparting\s+from\b", 0.93),
    _r("departed_from", r"\bdepart(?:s)?\s+from\s+(?:the\s+)?case[-\s]law\b", 0.93),
    # Followed / settled
    _r("followed", r"\bsettled\s+case[-\s]law\b", 0.92),
    _r("followed", r"\b(the\s+Court|it)\s+has\s+(?:consistently|repeatedly)\s+held\b", 0.92),
    _r("followed", r"\bas\s+the\s+Court\s+(?:held|ruled)\s+in\b", 0.9),
    _r("applied", r"\brestated\s+in\b", 0.88),
    _r("applied", r"\bapplied\s+(?:to|in)\b", 0.8),
    _r("considered", r"\bin\s+the\s+light\s+of\b", 0.65),
]

# EU judgments draw from both common-law verbs and CJEU-specific phrasing.
# Order: EU-specific first (more distinctive), then common-law fallback.
EU_RULES: list[_Rule] = EU_EXTRA_RULES + COMMON_LAW_RULES


register_rules("EU", EU_RULES)
