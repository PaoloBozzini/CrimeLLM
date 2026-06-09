"""Single-shot autofetch worker: process one queued job, return.

The loop lives in the CLI (``clg autofetch drain --max N``) so signal
handling, pacing, and termination policy stay near the operator. Keeping
``run_once`` a pure function over its dependencies makes every branch
trivially testable with in-memory doubles for ``Source``.

Branches:

- queue empty → ``IDLE``
- resolver returns ``None`` → terminal ``SKIPPED`` (no point retrying).
- breaker open → ``CIRCUIT_OPEN``; queue lease is rolled back so the
  attempt counter doesn't tick over an outage we know about.
- ``fetch_one`` / ``load`` raises → ``FAILED``; queue records the error
  and breaker counts a failure. Job returns to pending until
  ``max_attempts`` exhausted.
- success → ``OK``; queue marks done, breaker resets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

_log = logging.getLogger(__name__)

from ..ingest._base import IngestContext, Source
from .circuit_breaker import CircuitBreaker
from .exceptions import UnsupportedCite
from .queue import SqliteQueue
from .resolver import resolve


class JobOutcome(str, Enum):
    IDLE = "idle"
    OK = "ok"
    SKIPPED = "skipped"
    CIRCUIT_OPEN = "circuit_open"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class WorkerContext:
    queue: SqliteQueue
    breaker: CircuitBreaker
    sources: Mapping[str, Source]
    ingest_ctx: IngestContext
    max_attempts: int = 3
    # Phase E.2: cap recursive cite-walk depth after a successful fetch.
    # 0 disables cascade entirely (Phase A/B compatible). Reads from
    # ``Settings.autofetch_max_depth`` in production wiring.
    cascade_max_depth: int = 0


@dataclass(frozen=True, slots=True)
class RunResult:
    outcome: JobOutcome
    cite_id: str | None = None
    source: str | None = None
    error: str | None = None


def run_once(ctx: WorkerContext) -> RunResult:
    started = time.monotonic()
    job = ctx.queue.lease()
    if job is None:
        return RunResult(JobOutcome.IDLE)

    def _emit(result: RunResult) -> RunResult:
        # Single per-job log record. Attaches a structured ``job`` dict via
        # ``extra=`` so JSON formatters surface every field; printf-style
        # message stays scannable in plain-text logs.
        elapsed_ms = int((time.monotonic() - started) * 1000)
        payload = {
            "cite_id": job.cite_id,
            "source": result.source or job.source,
            "outcome": result.outcome.value,
            "attempts": job.attempts,
            "depth": job.depth,
            "duration_ms": elapsed_ms,
            "error": result.error,
        }
        _log.info(
            "autofetch %s cite=%s source=%s attempts=%d depth=%d in %dms",
            result.outcome.value,
            payload["cite_id"],
            payload["source"],
            payload["attempts"],
            payload["depth"],
            elapsed_ms,
            extra={"job": payload},
        )
        return result

    source_name = resolve(job.cite_id)
    if source_name is None:
        ctx.queue.mark_skipped(job.cite_id, "no resolver match")
        return _emit(RunResult(
            JobOutcome.SKIPPED, cite_id=job.cite_id, error="no resolver match"
        ))

    source = ctx.sources.get(source_name)
    if source is None:
        # Resolver knows the name but the worker wasn't given an instance.
        # Treat as skipped so we don't burn attempts on a config gap.
        ctx.queue.mark_skipped(job.cite_id, f"source '{source_name}' not configured")
        return _emit(RunResult(
            JobOutcome.SKIPPED,
            cite_id=job.cite_id,
            source=source_name,
            error=f"source '{source_name}' not configured",
        ))

    if not ctx.breaker.allow(source_name):
        ctx.queue.release(job.cite_id)
        return _emit(RunResult(
            JobOutcome.CIRCUIT_OPEN, cite_id=job.cite_id, source=source_name
        ))

    fetched_paths: dict[str, "Path"] = {}
    try:
        fetched_paths = source.fetch_one(ctx.ingest_ctx, job.cite_id) or {}
        # Phase E will chain parse/load/embed/link. For Phase B we only
        # exercise the worker boundary; the fake Source's ``load`` covers
        # the happy-path assertion path.
        source.load(ctx.ingest_ctx)
    except UnsupportedCite as exc:
        # Cite isn't broken — the source just can't fetch this shape. Don't
        # tick the breaker; don't retry. Persist a terminal skip with the
        # reason so an operator sees it in ``list-pending`` / ``status``.
        ctx.queue.mark_skipped(job.cite_id, str(exc))
        return _emit(RunResult(
            JobOutcome.SKIPPED,
            cite_id=job.cite_id,
            source=source_name,
            error=str(exc),
        ))
    except Exception as exc:  # noqa: BLE001 — third-party APIs raise anything
        ctx.queue.mark_failed(job.cite_id, str(exc), max_attempts=ctx.max_attempts)
        ctx.breaker.record_failure(source_name)
        return _emit(RunResult(
            JobOutcome.FAILED,
            cite_id=job.cite_id,
            source=source_name,
            error=str(exc),
        ))

    ctx.queue.mark_done(job.cite_id)
    ctx.breaker.record_success(source_name)

    # F.1: tag whatever node the loader just persisted with the
    # auto-ingested / unvalidated flags. Best-effort: missing store or
    # no-match returns 0; worker keeps going either way.
    store = getattr(ctx.ingest_ctx, "store", None)
    if store is not None:
        try:
            from .quarantine import mark_auto_ingested

            mark_auto_ingested(job.cite_id, store=store)
        except Exception:  # noqa: BLE001 — quarantine flip is best-effort
            pass

    if ctx.cascade_max_depth > 0 and fetched_paths:
        # Lazy import keeps the worker importable when cascade isn't wired.
        from .cascade import cascade_from_paths

        cascade_from_paths(
            fetched_paths.values(),
            parent_depth=job.depth,
            max_depth=ctx.cascade_max_depth,
            queue=ctx.queue,
        )

    return _emit(RunResult(JobOutcome.OK, cite_id=job.cite_id, source=source_name))
