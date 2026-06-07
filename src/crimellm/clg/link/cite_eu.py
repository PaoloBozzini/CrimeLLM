"""EU citation parser — hand-rolled regex over canonical identifiers.

EU primary law uses three identifier schemes side-by-side:

* **ECLI:EU** — case law (``ECLI:EU:C:2014:317``). ``C``=Court of Justice,
  ``T``=General Court, ``F``=Civil Service Tribunal (defunct).
* **CELEX** — universal sector code (``32016R0679`` = GDPR, ``61991CJ0267`` =
  *Keck Mithouard*). First digit is the sector (3 = legislation, 6 = case
  law); a single letter splits the document type.
* **ELI** — slash-path URIs (``eli/reg/2016/679``). Most useful when the
  source body publishes them inline; CELEX dominates citation prose.

Plus inline treaty article refs (``Article 101 TFEU``, ``Art. 6 TEU``) which
the firm uses constantly when working competition / fundamental rights.

Normalised IDs:

* ECLI / CELEX → verbatim (already canonical).
* ELI → ``eu/<type>/<year>/<num>`` with ``reg`` | ``dir`` | ``dec``.
* Treaty article → ``eu/treaty/<tfeu|teu>/article/<N>``.
"""

from __future__ import annotations

import re

from .cite_registry import CitationHit, register

JURISDICTION = "EU"

# ECLI:EU:C:2014:317 — case law.
_ECLI_EU_RE = re.compile(
    r"\bECLI:EU:(?P<court>[CTF]):\d{4}:\d+\b"
)

# 32016R0679, 32019L0770, 61991CJ0267 — CELEX. Sector → document kind:
#   3* = legislation (regulations / directives / decisions)
#   6* = case law
_CELEX_RE = re.compile(
    r"\b(?P<sector>[1-9])(?P<year>\d{4})(?P<type>[A-Z]{1,2})(?P<num>\d{4})\b"
)

# eli/reg/2016/679 — ELI legislative URI.
_ELI_RE = re.compile(
    r"\beli/(?P<type>reg|dir|dec|rec)/(?P<year>\d{4})/(?P<num>\d+)\b",
    re.IGNORECASE,
)

# Article 101 TFEU / Art. 6 TEU — treaty article references.
_TREATY_ART_RE = re.compile(
    r"\b(?:Article|Art\.?)\s+(?P<num>\d+[a-z]?)"
    r"(?:\((?P<para>\d+)\))?"
    r"\s+(?P<treaty>TFEU|TEU)\b",
    re.IGNORECASE,
)


def _celex_kind(sector: str) -> str:
    """3* = legislation → ``provision``; 6* = judgment → ``case``."""
    return "case" if sector == "6" else "provision"


class EuCitationParser:
    jurisdiction = JURISDICTION

    def extract(self, text: str) -> list[CitationHit]:
        if not text:
            return []
        hits: list[CitationHit] = []

        for m in _ECLI_EU_RE.finditer(text):
            hits.append(
                CitationHit(
                    raw=m.group(0),
                    normalised_id=m.group(0),
                    kind="case",
                    span=m.span(),
                    jurisdiction=JURISDICTION,
                )
            )

        for m in _CELEX_RE.finditer(text):
            hits.append(
                CitationHit(
                    raw=m.group(0),
                    normalised_id=m.group(0),
                    kind=_celex_kind(m["sector"]),
                    span=m.span(),
                    jurisdiction=JURISDICTION,
                )
            )

        for m in _ELI_RE.finditer(text):
            hits.append(
                CitationHit(
                    raw=m.group(0),
                    normalised_id=f"eu/{m['type'].lower()}/{m['year']}/{m['num']}",
                    kind="provision",
                    span=m.span(),
                    jurisdiction=JURISDICTION,
                )
            )

        for m in _TREATY_ART_RE.finditer(text):
            treaty = m["treaty"].lower()
            para = f"/para/{m['para']}" if m.group("para") else ""
            hits.append(
                CitationHit(
                    raw=m.group(0),
                    normalised_id=f"eu/treaty/{treaty}/article/{m['num'].lower()}{para}",
                    kind="provision",
                    span=m.span(),
                    jurisdiction=JURISDICTION,
                )
            )

        hits.sort(key=lambda h: h.span[0])
        return hits


register(EuCitationParser())
