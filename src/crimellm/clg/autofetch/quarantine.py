"""Quarantine: tag auto-loaded nodes, exclude from eval, support promotion.

Worker flow:

    fetch_one → load → mark_auto_ingested(cite_id, store=...)

The flip is a separate Cypher rather than a loader kwarg because each
``Source.load`` returns counts but not the ids it touched — re-running the
ID-keyed match after load is dead-simple and label-agnostic. Hand-loaded
nodes don't see this code path; they keep the dataclass defaults
(``auto_ingested=false, validated=true``).

Eval / retrieval boundaries (Phase F.2) read
:data:`PRESENCE_VALIDATED_CYPHER` so the unvalidated filter has one source
of truth.
"""

from __future__ import annotations

from typing import Any, Protocol


class _Store(Protocol):
    def run(self, cypher: str, **kwargs: Any) -> Any: ...


# Used by worker after a successful fetch+load. Label-agnostic so the
# resolver doesn't have to tell us whether a given cite_id resolves to a
# Case, Provision, or Instrument. ``ON MATCH`` is intentionally absent —
# the loader's ON CREATE sets the flags; this is the "flip after the fact"
# for hand-loaded nodes that the worker is now adopting as auto-ingested.
_MARK_AUTO_CYPHER = """
MATCH (n)
  WHERE (n:Case OR n:Provision OR n:Instrument) AND n.id = $cite_id
SET n.auto_ingested = true, n.validated = false
RETURN n.id AS id
"""


_PROMOTE_CYPHER = """
MATCH (n)
  WHERE (n:Case OR n:Provision OR n:Instrument) AND n.id = $cite_id
SET n.validated = true
RETURN n.id AS id
"""


# Filter expression for eval / retrieval boundaries. Built as a string so
# downstream callers can splice it into their own MATCH/WHERE clauses. The
# ``$include_unvalidated`` parameter (boolean) lets callers opt out per
# query — useful for the operator-facing autofetch CLI which wants to see
# the quarantined rows.
PRESENCE_VALIDATED_CYPHER = """
WITH coalesce($include_unvalidated, false) AS include_unvalidated
WHERE include_unvalidated OR coalesce(n.validated, true) = true
"""


def mark_auto_ingested(cite_id: str, *, store: _Store) -> int:
    """Flip ``auto_ingested=true, validated=false`` for any node with ``$cite_id``.

    Returns the number of rows touched (0 when the loader didn't end up
    persisting anything matching the cite id — common when a cite resolves
    to a doc whose internal id differs from the cite shape).
    """
    rows = list(store.run(_MARK_AUTO_CYPHER, cite_id=cite_id))
    return len(rows)


def promote(cite_id: str, *, store: _Store) -> int:
    """Flip ``validated=true`` for nodes matching ``$cite_id``.

    Returns the number of rows touched. ``auto_ingested`` is left alone so
    the historical fact "this was machine-loaded" survives — the
    ``validated`` flag is what gates the eval filter.
    """
    rows = list(store.run(_PROMOTE_CYPHER, cite_id=cite_id))
    return len(rows)
