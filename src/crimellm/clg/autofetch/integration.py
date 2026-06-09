"""Bridge: retrieval pipeline → autofetch queue.

Two entry points used by ``retrieval/query.py``:

* :func:`enqueue_missing_citations` — pure function over an explicit cite
  list + a store + an open queue. The unit-testable shape; lets callers
  inject a mock store and a tmp-path SQLite queue.
* :func:`enqueue_missing_for_query` — the high-level facade ``run_query``
  uses. Reads ``Settings.autofetch_enabled`` (no-op when disabled), opens
  the queue lazily, returns the list of newly-pending cite ids.

Kept deliberately small. The actual fetching, parsing, embedding lives in
the worker (``autofetch/worker.py``); this module only writes queue rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from ..config import Settings, get_settings
from .queue import SqliteQueue
from .resolver import resolve


class _Store(Protocol):
    """Minimum surface ``enqueue_missing_citations`` needs from a Neo4j store."""

    def run(self, cypher: str, **kwargs: Any) -> Iterable[dict[str, Any]]: ...


# Single MERGE-key check across Case / Provision / Instrument. The autofetch
# resolver doesn't (yet) tell us the entity kind for a given cite id, so we
# probe all three labels in one round-trip. Returns the matched ids.
_PRESENCE_CYPHER = """
UNWIND $ids AS id
OPTIONAL MATCH (n)
  WHERE (n:Case OR n:Provision OR n:Instrument) AND n.id = id
WITH id, n WHERE n IS NOT NULL
RETURN DISTINCT id AS id
"""


def enqueue_missing_citations(
    citations: Iterable[str],
    *,
    store: _Store,
    queue: SqliteQueue,
) -> list[str]:
    """Return the subset of ``citations`` newly enqueued for autofetch.

    Skips cites that:
    - already have a matching node in Neo4j (no need to fetch);
    - the resolver can't route to a source (no use queueing — the worker
      would mark_skipped on first lease anyway).

    Duplicates are dropped before the Neo4j round-trip.
    """
    unique = list(dict.fromkeys(c for c in citations if c))
    if not unique:
        return []

    rows = list(store.run(_PRESENCE_CYPHER, ids=unique))
    present = {r["id"] for r in rows}

    pending: list[str] = []
    for cite_id in unique:
        if cite_id in present:
            continue
        source = resolve(cite_id)
        if source is None:
            continue
        queue.enqueue(cite_id, source)
        pending.append(cite_id)
    return pending


def enqueue_missing_for_query(
    query: object,
    *,
    store: _Store,
    settings: Settings | None = None,
    queue_factory: Callable[[Path], SqliteQueue] | None = None,
) -> list[str]:
    """Facade used by ``run_query``.

    Returns the list of cite ids newly added to the queue. Returns ``[]``
    immediately when ``autofetch_enabled`` is false — caller can still
    expose ``Answer.pending_citations`` from the queue's existing rows if
    we ever decide to surface in-flight work, but Phase D keeps the live
    answer untouched when the flag is off.

    ``queue_factory`` is an injection seam so tests can pass an already-open
    in-memory queue. Production uses the default factory (open a fresh
    SqliteQueue at ``settings.autofetch_queue_path``).
    """
    settings = settings or get_settings()
    if not settings.autofetch_enabled:
        return []

    cites = list(getattr(query, "citations", []) or [])
    if not cites:
        return []

    factory = queue_factory or (lambda path: SqliteQueue(path))
    queue = factory(settings.autofetch_queue_path)
    owns_queue = queue_factory is None
    try:
        return enqueue_missing_citations(cites, store=store, queue=queue)
    finally:
        if owns_queue:
            queue.close()
