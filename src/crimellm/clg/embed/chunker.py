"""Natural-unit chunker.

* ``chunk_provision`` — one chunk per Provision, falling back to
  ``max_chars``-wide windows with overlap if the section text exceeds the
  embedder's context comfortably.
* ``chunk_case`` — splits the judgment body on blank lines (paragraphs),
  then re-window if a paragraph is itself too long.
* ``iter_chunks`` — dispatcher that picks the right chunker by parent type.

Chunk ids are content hashes (``sha256(text)`` truncated to 32 hex chars).
That makes MERGE idempotent and de-duplicates identical passages across
sources for free.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator

from ..models import Case, Chunk, Provision

DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP = 200
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"\s+")


def _chunk_id(text: str) -> str:
    return "ch-" + hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _window(text: str, max_chars: int, overlap: int) -> list[str]:
    """Slide a window across long text, preserving word boundaries best-effort."""
    if max_chars <= 0:
        return [text]
    if len(text) <= max_chars:
        return [text]
    if overlap >= max_chars:
        overlap = max_chars // 4
    out: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            # back off to last whitespace before end so we don't break a word
            wb = text.rfind(" ", start + max_chars // 2, end)
            if wb > start:
                end = wb
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        if start <= 0:
            start = end
    return out


# --- Provision -------------------------------------------------------------


def chunk_provision(
    provision: Provision,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Yield one or more Chunks for a Provision.

    Most sections are short enough to map 1:1 to a Chunk. Long sections get
    windowed with character-count overlap so the embedder always sees a
    full unit.
    """
    body = _normalise(provision.text)
    if not body:
        return []
    parts = _window(body, max_chars, overlap)
    return [
        Chunk(
            id=_chunk_id(part),
            text=part,
            parent_id=provision.id,
            parent_type="Provision",
        )
        for part in parts
    ]


# --- Case ------------------------------------------------------------------


def chunk_case(
    case: Case,
    body_text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Yield Chunks for a Case body text (passed in because Case has no text).

    ``Case`` itself only stores metadata; the parser keeps the body text
    next to the Case. This chunker takes both so the parent_id is correct.
    Splits on blank lines first; very long paragraphs get the same windowing
    treatment as long provisions.
    """
    if not body_text:
        return []
    out: list[Chunk] = []
    for para in _PARAGRAPH_SPLIT.split(body_text):
        para = _normalise(para)
        if not para:
            continue
        for piece in _window(para, max_chars, overlap):
            out.append(
                Chunk(
                    id=_chunk_id(piece),
                    text=piece,
                    parent_id=case.id,
                    parent_type="Case",
                )
            )
    return out


# --- Dispatcher ------------------------------------------------------------


def iter_chunks(
    items: Iterable[object],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[Chunk]:
    """Pass an iterable of Provisions and/or (Case, body) tuples.

    Cases get bundled as ``(case, body_text)`` because Case carries no text
    on the dataclass — the body lives wherever the Case was parsed.
    """
    for item in items:
        if isinstance(item, Provision):
            yield from chunk_provision(item, max_chars=max_chars, overlap=overlap)
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], Case)
            and isinstance(item[1], str)
        ):
            yield from chunk_case(item[0], item[1], max_chars=max_chars, overlap=overlap)
        else:  # pragma: no cover — guard against accidental misuse
            raise TypeError(
                f"iter_chunks expected Provision or (Case, body); got {type(item).__name__}"
            )
