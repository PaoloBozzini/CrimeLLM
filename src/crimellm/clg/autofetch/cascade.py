"""Walk newly-fetched docs, enqueue their cite-ids at the next depth.

The worker calls :func:`cascade_from_paths` after a successful
``Source.fetch_one`` so the autofetch graph snowballs outward from the
operator's seed cite. Cap at ``Settings.autofetch_max_depth`` to keep the
queue from exploding when a single doc cites hundreds of others.

Cite extraction reuses ``autofetch.resolver.scan_text`` (ELI / CourtListener
shapes) and ``link.cite_registry.extract_all`` (ECLI / CELEX / Ufr / etc.)
— same dispatch as ``parse_query`` so what gets enqueued for a typed query
matches what gets enqueued by cascade.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Side-effect imports register the per-jurisdiction citation parsers.
from ..link import cite_dk as _cite_dk  # noqa: F401
from ..link import cite_eu as _cite_eu  # noqa: F401
from ..link import cite_us as _cite_us  # noqa: F401
from ..link.cite_registry import extract_all as _extract_citations
from .queue import SqliteQueue
from .resolver import resolve, scan_text


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        # Binary or unreadable: caller treats as "no cites found".
        return ""


def _cites_in(text: str) -> list[str]:
    """Stable-dedup union of prose-parser hits + autofetch-shape scan."""
    seen: set[str] = set()
    out: list[str] = []
    for h in _extract_citations(text):
        if h.normalised_id and h.normalised_id not in seen:
            out.append(h.normalised_id)
            seen.add(h.normalised_id)
    for hit in scan_text(text):
        if hit not in seen:
            out.append(hit)
            seen.add(hit)
    return out


def cascade_from_paths(
    paths: Iterable[Path],
    *,
    parent_depth: int,
    max_depth: int,
    queue: SqliteQueue,
) -> list[str]:
    """Enqueue cites discovered in ``paths`` at ``parent_depth + 1``.

    Returns the list of cite ids actually inserted (queue dedup means
    already-present ids are silently skipped). When the next depth would
    exceed ``max_depth``, the whole call is a no-op — no point reading
    files we won't queue from.
    """
    child_depth = parent_depth + 1
    if child_depth > max_depth:
        return []

    seen: set[str] = set()
    newly_inserted: list[str] = []
    for path in paths:
        text = _read_text(path)
        if not text:
            continue
        for cite_id in _cites_in(text):
            if cite_id in seen:
                continue
            seen.add(cite_id)
            source = resolve(cite_id)
            if source is None:
                continue
            if queue.enqueue(cite_id, source, depth=child_depth):
                newly_inserted.append(cite_id)
    return newly_inserted
