"""Locate a citation in opinion text + extract the citing sentence.

eyecite gives us the citation's character span. We expand to sentence
boundaries on either side, cap the length, and hand the result to the
classifier. When eyecite isn't available or the opinion text is empty,
``extract_citing_sentence`` returns ``""`` — the cascade keeps working
(the rule + distilled tiers abstain; the LLM tiers see no context and tend
to fall back to ``neutral``).
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _expand_to_sentence(text: str, span_start: int, span_end: int) -> str:
    """Grow ``[span_start, span_end]`` outwards to nearest sentence breaks."""
    if not text:
        return ""
    span_start = max(0, span_start)
    span_end = min(len(text), span_end)

    # Walk backwards for a sentence start.
    head = text.rfind(".", 0, span_start)
    head = max(head, text.rfind("!", 0, span_start), text.rfind("?", 0, span_start))
    head = head + 1 if head >= 0 else 0

    # Walk forwards for a sentence end.
    rest = text[span_end:]
    m = _SENTENCE_END.search(rest)
    tail = span_end + m.start() if m else len(text)

    return _normalise(text[head:tail])


def extract_citing_sentence(
    opinion_text: str,
    *,
    cited_case_name: str = "",
    max_chars: int = 800,
) -> str:
    """Find the first mention of ``cited_case_name`` (or any citation-shaped
    fragment) in ``opinion_text`` and return the sentence around it.

    Pure-Python fallback when eyecite isn't installed. Good enough for the
    rules + distilled tiers; the LLM tiers benefit from eyecite's precision
    when available.
    """
    if not opinion_text:
        return ""

    # Try eyecite first; it understands legal citation strings better than
    # any regex we'd hand-write.
    try:
        from eyecite import get_citations  # type: ignore

        citations = get_citations(opinion_text)
        for cite in citations:
            span = getattr(cite, "span", None)
            if span is None or len(span) != 2:
                continue
            start, end = span
            return _expand_to_sentence(opinion_text, start, end)[:max_chars]
    except Exception:  # noqa: BLE001 — eyecite missing or fails
        pass

    if cited_case_name:
        idx = opinion_text.lower().find(cited_case_name.lower())
        if idx >= 0:
            return _expand_to_sentence(opinion_text, idx, idx + len(cited_case_name))[:max_chars]

    return ""
