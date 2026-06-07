"""US citation parser — thin wrapper around eyecite.

eyecite understands US reporter cites (``347 U.S. 483``, ``410 F.3d 442``,
etc.) plus short-form / id / supra back-references. We surface only
``FullCaseCitation`` (id/supra are pointers to prior cites; the loader
will resolve them after the initial pass).

If eyecite isn't installed, the parser registers but ``extract`` returns
an empty list. Removing US from ``enabled_jurisdictions`` skips even the
construction path.
"""

from __future__ import annotations

from .cite_registry import CitationHit, register

JURISDICTION = "US"


class UsCitationParser:
    jurisdiction = JURISDICTION

    def __init__(self) -> None:
        try:
            from eyecite import get_citations  # type: ignore

            self._get_citations = get_citations
        except ImportError:
            self._get_citations = None

    def extract(self, text: str) -> list[CitationHit]:
        if not text or self._get_citations is None:
            return []
        try:
            cites = self._get_citations(text)
        except Exception:  # noqa: BLE001 — eyecite failures shouldn't crash ingest
            return []

        out: list[CitationHit] = []
        for c in cites:
            if type(c).__name__ != "FullCaseCitation":
                continue
            raw = c.matched_text() if hasattr(c, "matched_text") else str(c)
            span = c.span() if callable(getattr(c, "span", None)) else getattr(c, "span", None)
            if span is None or len(span) != 2:
                continue
            corrected = (
                c.corrected_citation()
                if hasattr(c, "corrected_citation")
                else raw
            )
            out.append(
                CitationHit(
                    raw=raw,
                    normalised_id=corrected,
                    kind="case",
                    span=(int(span[0]), int(span[1])),
                    jurisdiction=JURISDICTION,
                )
            )
        return out


register(UsCitationParser())
