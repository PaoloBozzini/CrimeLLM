"""Parse the user's free-text question into a structured ``Query``.

Three knobs come out:

* ``jurisdiction`` — defaults to ``None`` (cross-jurisdiction). Heuristics
  bump it to ``US`` / ``UK`` / ``EW`` / ``EU`` / ``DK`` when the question
  makes it obvious ("U.S. Code §...", "Fraud Act 2006 s.2", "straffelovens
  § 279", "Article 101 TFEU"). CLI ``--jurisdiction`` always wins over
  inference. Inferred jurisdictions not in ``enabled_jurisdictions`` are
  cleared back to ``None`` so a disabled corpus doesn't accidentally drive
  retrieval.
* ``as_of`` — defaults to today (UTC). An explicit ISO date anywhere in the
  prompt ("as of 2018-05-12") overrides. CLI ``--as-of`` always wins.
* ``language`` — ISO 639-1 of the question body. Drives synthesis prompt
  language (Phase 8). Detection lives in :mod:`crimellm.common.language`
  (multi-signal: diacritics + stopwords + bigrams + suffixes). Defaults
  to ``"en"`` when undetermined — Claude handles EN > DA, so EN is the
  safer fallback.

We *don't* extract entities here — that's the job of seed + expand. Parsing
stays small so it can be reasoned about + tested cheaply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from ...common.language import detect_language
from ..models import Jurisdiction

__all__ = ["Jurisdiction", "Query", "parse_query", "detect_language"]

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# Cue phrases that bias jurisdiction. Conservative: when in doubt, no bias.
_US_CUES: tuple[str, ...] = (
    "u.s.c",
    "usc",
    "us code",
    "u.s. code",
    "scotus",
    "federal court",
    "circuit court",
    "district court",
    "ninth circuit",
    "second circuit",
    "supreme court of the united states",
    "courtlistener",
)
_UK_CUES: tuple[str, ...] = (
    "uk",
    "united kingdom",
    "england",
    "wales",
    "ukpga",
    "ukla",
    "ewca",
    "ewhc",
    "uksc",
    "privy council",
    "legislation.gov.uk",
    "fraud act",
    "theft act",
    "bribery act",
    "modern slavery act",
)
# Danish primary-law cues: portal name, doc-type prefixes, court tiers,
# named statutes, reporter shorthands, ECLI scheme. Lower-cased before
# substring match. Genitive forms ("straffelovens") substring-hit the bare
# stem ("straffeloven") so listing both isn't necessary.
_DK_CUES: tuple[str, ...] = (
    "retsinformation",
    "retsinformation.dk",
    "lovbekendtgørelse",
    "lbk nr",
    "bek nr",
    "lbk ",
    "bek ",
    "højesteret",
    "landsret",
    "østre landsret",
    "vestre landsret",
    "byret",
    "straffeloven",
    "aftaleloven",
    "markedsføringsloven",
    "databeskyttelsesloven",
    "erstatningsansvarsloven",
    "købeloven",
    "forbrugeraftaleloven",
    "forvaltningsloven",
    "udlændingeloven",
    "retsplejeloven",
    "selskabsloven",
    "u.20",
    "u.19",
    "fed ",
    "tfk ",
    "mad ",
    "ecli:dk:",
)
# EU primary-law cues: EUR-Lex / CELLAR portal names, treaty abbreviations,
# institution names (EN + DA), CELEX/ECLI schemes, common acts.
_EU_CUES: tuple[str, ...] = (
    "eur-lex",
    "eurlex",
    "cellar",
    "cjeu",
    "court of justice of the european union",
    "european court of justice",
    "general court",
    "tfeu",
    "tfeuf",
    "teu",
    "gdpr",
    "ecli:eu:",
    "regulation (eu)",
    "regulation (ec)",
    "directive (eu)",
    "direktiv",
    "forordning",
    "kommissionen",
    "rådet",
    "europa-parlamentet",
    "european parliament",
    "european commission",
    "council of the european union",
)

_CUE_TABLE: tuple[tuple[Jurisdiction, tuple[str, ...]], ...] = (
    ("US", _US_CUES),
    ("UK", _UK_CUES),
    ("EU", _EU_CUES),
    ("DK", _DK_CUES),
)


@dataclass(slots=True)
class Query:
    """Structured user query."""

    raw: str
    jurisdiction: Jurisdiction | None
    as_of: date
    language: str = "en"
    language_confidence: float = 0.0

    def with_overrides(
        self,
        *,
        jurisdiction: Jurisdiction | None = None,
        as_of: date | str | None = None,
        language: str | None = None,
    ) -> Query:
        new_as_of = self.as_of
        if as_of is not None:
            new_as_of = (
                as_of
                if isinstance(as_of, date)
                else datetime.strptime(as_of[:10], "%Y-%m-%d").date()
            )
        return Query(
            raw=self.raw,
            jurisdiction=jurisdiction if jurisdiction is not None else self.jurisdiction,
            as_of=new_as_of,
            language=language if language is not None else self.language,
            language_confidence=self.language_confidence,
        )


def _infer_jurisdiction(text: str) -> Jurisdiction | None:
    """Pick the jurisdiction whose cue list scores the most hits.

    Ties (incl. all-zero) return ``None`` so a cross-jurisdiction question
    or a question with no cues stays uncategorised. The CLI / caller can
    always override via ``Query.with_overrides(jurisdiction=...)``.
    """
    lower = text.lower()
    scores: list[tuple[Jurisdiction, int]] = [
        (code, sum(1 for cue in cues if cue in lower))
        for code, cues in _CUE_TABLE
    ]
    top_code, top_score = max(scores, key=lambda kv: kv[1])
    if top_score == 0:
        return None
    # Tie at the top → ambiguous → no bias.
    if sum(1 for _, s in scores if s == top_score) > 1:
        return None
    return top_code


def _infer_as_of(text: str) -> date:
    m = _ISO_DATE_RE.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return date.today()


def parse_query(text: str, *, settings: object | None = None) -> Query:
    """Heuristic parse. Cheap, deterministic, easy to override at the CLI.

    ``settings`` defaults to ``crimellm.clg.config.get_settings()`` so the
    enabled-jurisdiction filter applies automatically; tests / library
    callers can inject a custom Settings instance.
    """
    inferred = _infer_jurisdiction(text)

    # Enabled-jurisdiction filter: if the inferred jurisdiction isn't in
    # the active set, clear it. Operator overrides via with_overrides
    # always win, so CLI ``--jurisdiction DK`` still works even with DK
    # disabled (caller-knows-best).
    if inferred is not None:
        if settings is None:
            from ..config import get_settings

            settings = get_settings()
        is_enabled = getattr(settings, "is_enabled", None)
        if callable(is_enabled) and not is_enabled(inferred):
            inferred = None

    lang, lang_conf = detect_language(text)

    return Query(
        raw=text.strip(),
        jurisdiction=inferred,
        as_of=_infer_as_of(text),
        language=lang,
        language_confidence=lang_conf,
    )
