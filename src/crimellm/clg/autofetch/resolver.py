"""Canonical cite-id → source-name dispatch.

The autofetch worker hands the queue's ``cite_id`` to ``resolve(...)`` to
learn which Source impl owns the fetch. Per-source ``fetch_one`` then
re-parses the id into whatever shape its API needs (so the resolver stays
ignorant of API URL templates).

Patterns are matched in registration order; the first match wins. Adding a
new source = append one regex + source-name pair. ``None`` is a deliberate
return value — the worker treats unrecognised cites as ``status='skipped'``
rather than failing the queue.

Sources not in v1 (per ``docs/self-management-autofetch.local.md`` §4):
- ``ECLI:DK:*`` — domstol.dk scraping investigation deferred.
- ``U.YYYY.NNNNX`` — Karnov / Ufr subscription gated.
- ``[YYYY] EWCA Civ N`` — find_case_law neutral-cite → URL lookup deferred.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Rule:
    pattern: re.Pattern[str]
    source: str


# Order matters: more specific patterns first. The DK ELI ``eli/lov/...``
# rule must precede a generic ``eli/...`` rule if we ever add one.
_RULES: list[_Rule] = [
    # DK statutes via Retsinformation REST.
    _Rule(re.compile(r"^eli/(lov|lbk|bek|cir)/\d{4}/\d+(?:/.*)?$"), "retsinformation"),
    _Rule(re.compile(r"^DK/[a-zæøå]+/section/\d+(?:/.*)?$"), "retsinformation"),
    # EU CELEX — sector digit + 4-digit year + type letters + 4-digit num.
    # Match the bounded shape so unrelated 9-char digit blobs don't slip through.
    _Rule(re.compile(r"^[1-9]\d{4}[A-Z]{1,2}\d{4}$"), "eurlex"),
    # EU case law via ECLI.
    _Rule(re.compile(r"^ECLI:EU:[CTF]:\d{4}:\d+$"), "eurlex"),
    # EU ELI.
    _Rule(re.compile(r"^eu/(reg|dir|dec)/\d{4}/\d+(?:/.*)?$"), "eurlex"),
    # UK legislation ELI.
    _Rule(re.compile(r"^uk/(ukpga|asp|nia|wsi)/\d{4}/\d+(?:/.*)?$"), "legislation_uk"),
    # CourtListener opinion handles. Keeping the prefix explicit because raw
    # numeric ids would collide with anything else numeric we add later.
    _Rule(re.compile(r"^courtlistener:(opinion|cluster):\d+$"), "courtlistener"),
]


def resolve(cite_id: str) -> str | None:
    """Return the source name responsible for ``cite_id``, or ``None``."""
    if not cite_id:
        return None
    for rule in _RULES:
        if rule.pattern.match(cite_id):
            return rule.source
    return None


def register_rule(pattern: str | re.Pattern[str], source: str) -> None:
    """Append a custom rule. Useful for tests and Phase C source additions.

    Rules registered later have lower priority — the first match wins, so
    new sources should be more specific than existing patterns or they'll be
    shadowed.
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    _RULES.append(_Rule(compiled, source))


def rules() -> Iterable[tuple[str, str]]:
    """Snapshot the active dispatch table — ``(pattern, source)`` pairs."""
    return tuple((r.pattern.pattern, r.source) for r in _RULES)


# Substring-scan patterns (unanchored) for shapes the per-jurisdiction
# citation parsers don't cover: ELI slash-forms (DK/EU/UK), CourtListener
# opinion handles. Used by ``retrieval/parse_query.py`` to find canonical
# ids in user question text that the link-layer parsers would otherwise
# miss because they target prose-shaped citations.
_SCAN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\beli/(?:lov|lbk|bek|cir)/\d{4}/\d+\b"),
    re.compile(r"\beu/(?:reg|dir|dec)/\d{4}/\d+\b"),
    re.compile(r"\buk/(?:ukpga|asp|nia|wsi)/\d{4}/\d+\b"),
    re.compile(r"\bcourtlistener:(?:opinion|cluster):\d+\b"),
)


def scan_text(text: str) -> list[str]:
    """Pull autofetch-shape cite ids out of free text. Stable-dedup, in order."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for pat in _SCAN_PATTERNS:
        for m in pat.finditer(text):
            hit = m.group(0)
            if hit not in seen:
                out.append(hit)
                seen.add(hit)
    return out
