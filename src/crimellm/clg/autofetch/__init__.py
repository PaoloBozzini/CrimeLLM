"""Self-management subsystem: fetch missing-citation docs on demand.

When a citation in a user query or retrieved doc resolves to a canonical ID
(ECLI / ELI / CELEX / neutral cite / reporter triple) with no matching node
in Neo4j, the autofetch worker pulls the doc from its source, ingests it,
embeds it, links it, and tags it ``auto_ingested=true, validated=false`` so
it stays out of the eval gold set until a human promotes it.

Phase A creates this package shell. The queue (``queue.py``), worker
(``worker.py``), resolver (``resolver.py``), circuit breaker
(``circuit_breaker.py``), cascade (``cascade.py``), and quarantine
(``quarantine.py``) modules land in phases B–F.

See ``docs/self-management-autofetch.local.md`` for the full design.

Gated by ``Settings.autofetch_enabled`` (default ``False``). When disabled,
every entry point in this package is a no-op so production paths can import
it freely.
"""

from __future__ import annotations

__all__: list[str] = []
