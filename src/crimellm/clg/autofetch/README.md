# clg/autofetch

Background reconciliation of citations that miss the Neo4j graph.

When a query or a retrieved doc cites a law/case the graph doesn't have, the
worker fetches it from its source, parses it, embeds it, links it, and tags
it `auto_ingested=true, validated=false`. Sync query path stays fast: misses
are enqueued and the answer carries a `pending_citations` flag.

## Layout (phased)

| File | Phase | Purpose |
|---|---|---|
| `__init__.py` | A | Package shell |
| `queue.py` | B | `SqliteQueue` — enqueue / lease / mark_done / reclaim_stale_leases |
| `circuit_breaker.py` | B | Per-source backoff (token bucket + open/closed state) |
| `resolver.py` | B | `cite_id → (source_name, fetch_params)` dispatch |
| `worker.py` | B | Poll loop: lease → fetch → parse → load → mark_done |
| `cascade.py` | E | Walk new-doc cites → enqueue with `depth+1` (capped) |
| `quarantine.py` | F | Tag / list / promote auto-ingested docs |

## Config

All knobs in `clg/config.py` under `autofetch_*`. Default `enabled=False`.

## Design doc

`docs/self-management-autofetch.local.md` — architecture, phase plan,
SQLite schema, failure modes.

## Why SQLite (not Neo4j)

Queue must survive Neo4j rebuilds and eval freezes. WAL mode handles
multi-worker concurrency. Trivially inspectable (`sqlite3 data/autofetch.db`).
