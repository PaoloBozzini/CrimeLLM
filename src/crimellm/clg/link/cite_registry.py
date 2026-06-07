"""Per-jurisdiction citation-parser registry.

Each ``CitationParser`` implementation handles one jurisdiction's citation
grammar (US reporter cites via eyecite, DK Ufr + ECLI:DK, EU CELEX/ECLI/ELI).
Parsers self-register at import time; the link CLI dispatches by querying
``parsers_for_enabled(settings)`` and running every active parser over each
opinion.

Removing a jurisdiction = drop it from ``settings.enabled_jurisdictions``.
The parser module stays on disk; it's just skipped at dispatch.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..config import Settings

CitationKind = Literal["case", "provision"]


@dataclass(slots=True, frozen=True)
class CitationHit:
    """One citation located in an opinion / preamble.

    ``raw`` keeps the exact substring matched in the source; ``normalised_id``
    is the canonical identifier the graph loader will MERGE on (ECLI / ELI /
    reporter triple). ``jurisdiction`` is filled in by the registry on
    dispatch, not by the parser, so a parser can't accidentally tag hits
    with the wrong jurisdiction code.
    """

    raw: str
    normalised_id: str
    kind: CitationKind
    span: tuple[int, int]
    jurisdiction: str


@runtime_checkable
class CitationParser(Protocol):
    """Stateless. ``extract`` returns hits in document order."""

    jurisdiction: str

    def extract(self, text: str) -> list[CitationHit]: ...


_REGISTRY: dict[str, CitationParser] = {}


def register(parser: CitationParser) -> None:
    """Register ``parser`` for its declared jurisdiction.

    Last-write-wins so a test or downstream caller can swap in a mock without
    monkey-patching. Asserts the parser tags hits with the same jurisdiction
    it claims; mismatches are silent footguns otherwise.
    """
    code = parser.jurisdiction.upper()
    _REGISTRY[code] = parser


def unregister(jurisdiction: str) -> CitationParser | None:
    """Drop the registered parser for ``jurisdiction`` if present."""
    return _REGISTRY.pop(jurisdiction.upper(), None)


def for_jurisdiction(jurisdiction: str) -> CitationParser | None:
    return _REGISTRY.get(jurisdiction.upper())


def all_parsers() -> list[CitationParser]:
    return list(_REGISTRY.values())


def registered_jurisdictions() -> list[str]:
    return sorted(_REGISTRY)


def parsers_for_enabled(settings: Settings) -> list[CitationParser]:
    """Return the registered parsers whose jurisdiction is enabled in config."""
    enabled = {j.upper() for j in settings.enabled_jurisdictions}
    return [p for code, p in _REGISTRY.items() if code in enabled]


def extract_all(
    text: str,
    *,
    parsers: Iterable[CitationParser] | None = None,
    settings: Settings | None = None,
) -> list[CitationHit]:
    """Run every active parser over ``text``; return all hits.

    Hits are tagged with the parser's jurisdiction. No dedup — different
    parsers don't overlap citation grammars in practice, and a downstream
    loader is the right place to MERGE by ``normalised_id`` anyway.
    """
    if parsers is None:
        if settings is None:
            parsers = all_parsers()
        else:
            parsers = parsers_for_enabled(settings)
    hits: list[CitationHit] = []
    for p in parsers:
        hits.extend(p.extract(text))
    return hits
