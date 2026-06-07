"""Danish citation parser — hand-rolled regex.

Covers the vocab a Danish law firm actually pastes into a research prompt:

* **Ufr** — Ugeskrift for Retsvæsen. ``U.YYYY.NNNNX`` where ``X`` is the
  court suffix (H=Højesteret, V=Vestre Landsret, Ø=Østre Landsret, B=Byret).
* **Reporter cites** — ``FED YYYY.NNNN`` (Forsikrings- og Erstatningsret),
  ``TfK YYYY.NNNN`` (Kriminalret), ``MAD YYYY.NNNN`` (Miljø).
* **ECLI:DK** — ``ECLI:DK:HR:2023:123`` and similar. Canonical id already.
* **Statute references** — ``straffelovens § 279 stk. 2`` and friends. The
  named-statute list is small + stable; expanded in Phase 4 once
  Retsinformation ingest is wired up.

Hits use ``normalised_id`` shaped to match downstream graph keys:

* Ufr → ``U.YYYY.NNNN.X`` (period-delimited)
* Reporter → ``<REPORTER>.YYYY.NNNN``
* ECLI:DK → verbatim
* Statute → ``DK/<short_title>/section/<N>[/(stk).M][/(nr).K]``
"""

from __future__ import annotations

import re

from .cite_registry import CitationHit, register

JURISDICTION = "DK"

# Statute short titles the firm commonly cites. Genitive form (`-s` suffix)
# is the usual surface in Danish prose so both bare + genitive match.
_DK_STATUTE_SHORT_TITLES: tuple[str, ...] = (
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
)
_STATUTE_ALT = "|".join(t + "s?" for t in _DK_STATUTE_SHORT_TITLES)

# U.2010.1234H — Ufr citation; suffix is court code.
_UFR_RE = re.compile(
    r"\bU\.(?P<year>\d{4})\.(?P<num>\d{1,5})(?P<court>[HVØB])\b"
)

# FED|TfK|MAD YYYY.NNNN — specialist reporters.
_REPORTER_RE = re.compile(
    r"\b(?P<reporter>FED|TfK|MAD)\s+(?P<year>\d{4})\.(?P<num>\d{1,5})\b"
)

# ECLI:DK:<COURT>:YYYY:<id> — Danish ECLI scheme.
_ECLI_DK_RE = re.compile(
    r"\bECLI:DK:[A-Z]+:\d{4}:[A-Z0-9.]+\b"
)

# straffelovens § 279 stk. 2 nr. 1 — named statute + section path.
_STATUTE_RE = re.compile(
    r"\b(?P<title>" + _STATUTE_ALT + r")"
    r"\s+§\s*(?P<section>\d+[a-z]?)"
    r"(?:\s*,?\s*stk\.\s*(?P<stk>\d+))?"
    r"(?:\s*,?\s*nr\.\s*(?P<nr>\d+))?",
    re.IGNORECASE,
)


def _short_title(matched: str) -> str:
    """Strip the genitive ``-s`` so ``straffelovens`` → ``straffeloven``."""
    base = matched.lower()
    return base[:-1] if base.endswith("s") and base[:-1] in _DK_STATUTE_SHORT_TITLES else base


def _statute_id(title: str, section: str, stk: str | None, nr: str | None) -> str:
    parts = [f"DK/{_short_title(title)}/section/{section.lower()}"]
    if stk:
        parts.append(f"stk.{stk}")
    if nr:
        parts.append(f"nr.{nr}")
    return "/".join(parts)


class DkCitationParser:
    jurisdiction = JURISDICTION

    def extract(self, text: str) -> list[CitationHit]:
        if not text:
            return []
        hits: list[CitationHit] = []

        for m in _UFR_RE.finditer(text):
            hits.append(
                CitationHit(
                    raw=m.group(0),
                    normalised_id=f"U.{m['year']}.{m['num']}.{m['court']}",
                    kind="case",
                    span=m.span(),
                    jurisdiction=JURISDICTION,
                )
            )

        for m in _REPORTER_RE.finditer(text):
            hits.append(
                CitationHit(
                    raw=m.group(0),
                    normalised_id=f"{m['reporter']}.{m['year']}.{m['num']}",
                    kind="case",
                    span=m.span(),
                    jurisdiction=JURISDICTION,
                )
            )

        for m in _ECLI_DK_RE.finditer(text):
            hits.append(
                CitationHit(
                    raw=m.group(0),
                    normalised_id=m.group(0),
                    kind="case",
                    span=m.span(),
                    jurisdiction=JURISDICTION,
                )
            )

        for m in _STATUTE_RE.finditer(text):
            hits.append(
                CitationHit(
                    raw=m.group(0),
                    normalised_id=_statute_id(
                        m["title"], m["section"], m.group("stk"), m.group("nr")
                    ),
                    kind="provision",
                    span=m.span(),
                    jurisdiction=JURISDICTION,
                )
            )

        hits.sort(key=lambda h: h.span[0])
        return hits


register(DkCitationParser())
