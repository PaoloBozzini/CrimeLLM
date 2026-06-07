"""Tier 1 — Danish cue phrases for Højesteret / Landsret / Byret citations.

Civil-law (DK) judges write more compact treatment language than common-law
opinions. The vocabulary collapses to a handful of fixed phrases:

* ``i overensstemmelse med`` — "in accordance with" → followed
* ``stadfæstet af`` / ``tiltrædes`` — affirmed
* ``fraviger`` / ``har fraveget`` — departed_from (no formal overrule in DK)
* ``kritiseret af`` — criticised
* ``tilsidesat`` / ``forkastes`` — not_followed / set aside
* ``fortolket`` / ``henvises til`` — considered
* ``jf.`` — informal "cf." → considered (low confidence)

DK doesn't formally ``overrule`` prior judgments — Højesteret can depart
from practice without overturning the earlier opinion. So the
``overruled`` / ``reversed`` common-law labels are intentionally absent
from this list.
"""

from __future__ import annotations

import re

from .treatment_base import TreatmentLabel
from .treatment_rules import _Rule, register_rules


def _ci(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _r(label: TreatmentLabel, pattern: str, confidence: float) -> _Rule:
    return _Rule(label=label, pattern=_ci(pattern), confidence=confidence)


# Ordered adverse-first so the most decisive cue wins in mixed sentences.
DK_RULES: list[_Rule] = [
    # Departure / criticism / not-followed (civil-law adverse signals)
    _r("departed_from", r"\b(fraviger|har\s+fraveget|fravig(?:es|t))\b", 0.95),
    _r("criticised", r"\bkritiseret\s+(?:af|i)\b", 0.93),
    _r("criticised", r"\bafvigende\s+resultat\b", 0.85),
    _r("not_followed", r"\btilsidesat\b", 0.92),
    _r("not_followed", r"\bforkastes\b", 0.9),
    _r("not_followed", r"\b(?:Højesteret|landsretten)\s+forkastede\b", 0.92),
    _r("distinguished", r"\b(?:adskiller|adskilt)\s+sig\s+fra\b", 0.9),
    _r("doubted", r"\brejser\s+tvivl\s+om\b", 0.88),
    # Favourable / neutral
    _r("affirmed", r"\bstadfæstet(?:\s+af)?\b", 0.93),
    _r("affirmed", r"\btiltrædes\b", 0.9),
    _r("followed", r"\bi\s+overensstemmelse\s+med\b", 0.9),
    _r("followed", r"\bunder\s+henvisning\s+til\b", 0.85),
    _r("applied", r"\b(anvend(?:er|te|t)|anvendes)\s+(?:reglen|princippet)\s+(?:i|fra)\b", 0.88),
    _r("considered", r"\bfortolket\b", 0.78),
    _r("considered", r"\bjf\.\s", 0.55),
    _r("considered", r"\bhenviser\s+til\b", 0.7),
]


register_rules("DK", DK_RULES)
