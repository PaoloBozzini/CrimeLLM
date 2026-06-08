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

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

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


@dataclass(frozen=True, slots=True)
class RunResult:
    outcome: JobOutcome
    cite_id: str | None = None
    source: str | None = None
    error: str | None = None


def run_once(ctx: WorkerContext) -> RunResult:
    job = ctx.queue.lease()
    if job is None:
        return RunResult(JobOutcome.IDLE)

    source_name = resolve(job.cite_id)
    if source_name is None:
        ctx.queue.mark_skipped(job.cite_id, "no resolver match")
        return RunResult(
            JobOutcome.SKIPPED, cite_id=job.cite_id, error="no resolver match"
        )

    source = ctx.sources.get(source_name)
    if source is None:
        # Resolver knows the name but the worker wasn't given an instance.
        # Treat as skipped so we don't burn attempts on a config gap.
        ctx.queue.mark_skipped(job.cite_id, f"source '{source_name}' not configured")
        return RunResult(
            JobOutcome.SKIPPED,
            cite_id=job.cite_id,
            source=source_name,
            error=f"source '{source_name}' not configured",
        )

    if not ctx.breaker.allow(source_name):
        ctx.queue.release(job.cite_id)
        return RunResult(
            JobOutcome.CIRCUIT_OPEN, cite_id=job.cite_id, source=source_name
        )

    try:
        source.fetch_one(ctx.ingest_ctx, job.cite_id)
        # Phase E will chain parse/load/embed/link. For Phase B we only
        # exercise the worker boundary; the fake Source's ``load`` covers
        # the happy-path assertion path.
        source.load(ctx.ingest_ctx)
    except UnsupportedCite as exc:
        # Cite isn't broken — the source just can't fetch this shape. Don't
        # tick the breaker; don't retry. Persist a terminal skip with the
        # reason so an operator sees it in ``list-pending`` / ``status``.
        ctx.queue.mark_skipped(job.cite_id, str(exc))
        return RunResult(
            JobOutcome.SKIPPED,
            cite_id=job.cite_id,
            source=source_name,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — third-party APIs raise anything
        ctx.queue.mark_failed(job.cite_id, str(exc), max_attempts=ctx.max_attempts)
        ctx.breaker.record_failure(source_name)
        return RunResult(
            JobOutcome.FAILED,
            cite_id=job.cite_id,
            source=source_name,
            error=str(exc),
        )

    ctx.queue.mark_done(job.cite_id)
    ctx.breaker.record_success(source_name)
    return RunResult(JobOutcome.OK, cite_id=job.cite_id, source=source_name)
