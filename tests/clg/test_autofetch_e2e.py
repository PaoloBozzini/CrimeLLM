"""Phase B.6: end-to-end queue → worker → done.

Stitches every B-phase component together with a fake ``Source`` so the
integration story is exercised without external APIs or Neo4j. Covers:

- Multi-cite enqueue → drain in FIFO order.
- Transient failure path with breaker tripping and recovering.
- Stale-lease reclaim after a "crashed" worker.
- Skipped cite (no resolver) does not block subsequent work.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from crimellm.clg.autofetch.circuit_breaker import CircuitBreaker
from crimellm.clg.autofetch.queue import SqliteQueue
from crimellm.clg.autofetch.resolver import register_rule
from crimellm.clg.autofetch.worker import JobOutcome, WorkerContext, run_once
from crimellm.clg.ingest._base import IngestContext, LoadReport, Source


class _ScriptedSource(Source):
    """Source whose fetch result is driven by a per-cite script of outcomes."""

    name = "scripted"

    def __init__(self) -> None:
        self.script: dict[str, list[BaseException | None]] = {}
        self.calls: list[str] = []

    def download(self, ctx: IngestContext) -> dict[str, Path]:  # pragma: no cover
        return {}

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:  # pragma: no cover
        yield from ()

    def load(self, ctx: IngestContext) -> LoadReport:
        return LoadReport(source=self.name, counts={"docs": 1})

    def supports_single_fetch(self) -> bool:
        return True

    def fetch_one(self, ctx: IngestContext, cite_id: str) -> dict[str, Path]:
        self.calls.append(cite_id)
        if cite_id in self.script and self.script[cite_id]:
            action = self.script[cite_id].pop(0)
            if isinstance(action, BaseException):
                raise action
        return {cite_id: Path(f"/tmp/{cite_id}")}


@pytest.fixture(autouse=True)
def _register_scripted_rule():
    register_rule(r"^scripted:[a-z0-9-]+$", "scripted")
    yield
    from crimellm.clg.autofetch import resolver as R

    R._RULES.pop()


@pytest.fixture
def setup(tmp_path: Path):
    queue = SqliteQueue(tmp_path / "q.db")
    breaker = CircuitBreaker(tmp_path / "q.db", failure_threshold=2, open_seconds=60)
    src = _ScriptedSource()
    ctx = WorkerContext(
        queue=queue,
        breaker=breaker,
        sources={"scripted": src},
        ingest_ctx=IngestContext(),
        max_attempts=3,
    )
    yield ctx, src
    queue.close()
    breaker.close()


# --- happy path ------------------------------------------------------------


def test_drain_processes_all_pending(setup) -> None:
    ctx, src = setup
    ctx.queue.enqueue("scripted:a", "scripted")
    ctx.queue.enqueue("scripted:b", "scripted")
    ctx.queue.enqueue("scripted:c", "scripted")

    results = [run_once(ctx) for _ in range(4)]
    outcomes = [r.outcome for r in results]

    assert outcomes == [JobOutcome.OK, JobOutcome.OK, JobOutcome.OK, JobOutcome.IDLE]
    # FIFO preserved.
    assert src.calls == ["scripted:a", "scripted:b", "scripted:c"]
    assert ctx.queue.list_pending() == []


# --- failure → recovery ---------------------------------------------------


def test_transient_failure_then_success(setup) -> None:
    ctx, src = setup
    src.script["scripted:flaky"] = [RuntimeError("503")]  # fail once, then succeed
    ctx.queue.enqueue("scripted:flaky", "scripted")

    r1 = run_once(ctx)
    assert r1.outcome == JobOutcome.FAILED

    r2 = run_once(ctx)
    assert r2.outcome == JobOutcome.OK
    assert ctx.queue.list_pending() == []


def test_breaker_opens_after_repeated_failures_then_recovers(setup) -> None:
    ctx, src = setup
    # Two distinct cites both fail twice — breaker threshold is 2.
    src.script["scripted:a"] = [RuntimeError("boom"), RuntimeError("boom")]
    src.script["scripted:b"] = [RuntimeError("boom"), RuntimeError("boom")]
    ctx.queue.enqueue("scripted:a", "scripted")
    ctx.queue.enqueue("scripted:b", "scripted")

    r1 = run_once(ctx)
    r2 = run_once(ctx)
    assert r1.outcome == JobOutcome.FAILED
    assert r2.outcome == JobOutcome.FAILED

    # Breaker now open — next run releases without fetching.
    r3 = run_once(ctx)
    assert r3.outcome == JobOutcome.CIRCUIT_OPEN
    assert src.calls.count("scripted:a") + src.calls.count("scripted:b") == 2

    # Force breaker cooldown to expire; next call transitions to half_open
    # and trial-allows. Drop the failure scripts so the trial succeeds.
    ctx.breaker._set_next_attempt_for_test(
        "scripted", datetime.now(timezone.utc) - timedelta(seconds=1)
    )
    src.script = {}

    r4 = run_once(ctx)
    assert r4.outcome == JobOutcome.OK


# --- stale lease -----------------------------------------------------------


def test_stale_lease_reclaim_unblocks_queue(setup) -> None:
    ctx, _ = setup
    ctx.queue.enqueue("scripted:stuck", "scripted")
    job = ctx.queue.lease()
    assert job is not None
    # Simulate worker crash: lease never released, but expires.
    ctx.queue._set_lease_until_for_test(
        "scripted:stuck", datetime.now(timezone.utc) - timedelta(minutes=10)
    )

    # Next worker picks the same job up via the same lease predicate.
    result = run_once(ctx)
    assert result.outcome == JobOutcome.OK


# --- skipped cite does not block ------------------------------------------


def test_skipped_cite_does_not_block_subsequent_work(setup) -> None:
    ctx, src = setup
    # Unknown shape: resolver returns None → terminal SKIPPED.
    ctx.queue.enqueue("garbage-cite", "unknown")
    ctx.queue.enqueue("scripted:ok", "scripted")

    r1 = run_once(ctx)
    r2 = run_once(ctx)
    assert {r1.outcome, r2.outcome} == {JobOutcome.SKIPPED, JobOutcome.OK}
    # The good cite still landed.
    assert "scripted:ok" in src.calls
