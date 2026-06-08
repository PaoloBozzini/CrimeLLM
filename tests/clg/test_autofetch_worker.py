"""Phase B.4: worker.run_once — process exactly one queued job.

Worker is a pure function over (queue, breaker, sources). Test doubles for
Source let us exercise every branch (success, transient fail, permanent
unknown-source skip, circuit-open release) without touching real APIs.

The CLI (B.5) drives the loop count: ``for _ in range(N): run_once(...)``.
Keeping the worker single-shot makes it trivially testable and lets the CLI
choose its own pacing / signal handling.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crimellm.clg.autofetch.circuit_breaker import CircuitBreaker
from crimellm.clg.autofetch.queue import SqliteQueue
from crimellm.clg.autofetch.resolver import register_rule
from crimellm.clg.autofetch.worker import JobOutcome, WorkerContext, run_once
from crimellm.clg.ingest._base import IngestContext, LoadReport, Source


# --- test doubles ----------------------------------------------------------


class _FakeSource(Source):
    name = "fake"

    def __init__(self, *, fetch_raises: Exception | None = None) -> None:
        self._fetch_raises = fetch_raises
        self.fetch_calls: list[str] = []
        self.load_calls = 0

    def download(self, ctx: IngestContext) -> dict[str, Path]:  # pragma: no cover
        return {}

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:  # pragma: no cover
        yield from ()

    def load(self, ctx: IngestContext) -> LoadReport:
        self.load_calls += 1
        return LoadReport(source=self.name, counts={"docs": 1})

    def supports_single_fetch(self) -> bool:
        return True

    def fetch_one(self, ctx: IngestContext, cite_id: str) -> dict[str, Path]:
        self.fetch_calls.append(cite_id)
        if self._fetch_raises is not None:
            raise self._fetch_raises
        return {cite_id: Path("/tmp/fake")}


# --- fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_fake_source_rule():
    """Map ``fake:*`` cite ids to the ``fake`` source for the duration of the test."""
    register_rule(r"^fake:[a-z0-9]+$", "fake")
    yield
    # The resolver has no public unregister hook (rules are append-only by
    # design — order matters). Drop the trailing rule we added so other
    # tests see a clean dispatch table.
    from crimellm.clg.autofetch import resolver as R

    R._RULES.pop()


@pytest.fixture
def queue(tmp_path: Path) -> SqliteQueue:
    q = SqliteQueue(tmp_path / "q.db")
    yield q
    q.close()


@pytest.fixture
def breaker(tmp_path: Path) -> CircuitBreaker:
    cb = CircuitBreaker(tmp_path / "q.db", failure_threshold=2, open_seconds=60)
    yield cb
    cb.close()


def _ctx(queue: SqliteQueue, breaker: CircuitBreaker, sources: dict[str, Source]) -> WorkerContext:
    return WorkerContext(
        queue=queue,
        breaker=breaker,
        sources=sources,
        ingest_ctx=IngestContext(),
        max_attempts=3,
    )


# --- outcomes --------------------------------------------------------------


def test_run_once_idle_when_queue_empty(
    queue: SqliteQueue, breaker: CircuitBreaker
) -> None:
    result = run_once(_ctx(queue, breaker, {}))
    assert result.outcome == JobOutcome.IDLE


def test_run_once_success_marks_done(
    queue: SqliteQueue, breaker: CircuitBreaker
) -> None:
    src = _FakeSource()
    queue.enqueue("fake:one", "fake")
    result = run_once(_ctx(queue, breaker, {"fake": src}))
    assert result.outcome == JobOutcome.OK
    assert result.cite_id == "fake:one"
    assert src.fetch_calls == ["fake:one"]
    assert src.load_calls == 1
    assert queue.list_pending() == []


def test_run_once_unknown_source_skips_terminally(
    queue: SqliteQueue, breaker: CircuitBreaker
) -> None:
    queue.enqueue("totally-unknown", "unknown")
    result = run_once(_ctx(queue, breaker, {}))
    assert result.outcome == JobOutcome.SKIPPED
    # Skipped jobs do not return to pending — they're terminal.
    assert queue.list_pending() == []


def test_run_once_circuit_open_releases_job(
    queue: SqliteQueue, breaker: CircuitBreaker
) -> None:
    breaker.record_failure("fake")
    breaker.record_failure("fake")  # threshold=2 → opens
    src = _FakeSource()
    queue.enqueue("fake:one", "fake")
    result = run_once(_ctx(queue, breaker, {"fake": src}))
    assert result.outcome == JobOutcome.CIRCUIT_OPEN
    # Job returns to pending without bumping attempts past the lease side
    # effect — breaker-open isn't the job's fault.
    pending = queue.list_pending()
    assert len(pending) == 1
    assert pending[0].cite_id == "fake:one"
    assert pending[0].attempts == 0  # attempt unwound
    assert src.fetch_calls == []


def test_run_once_fetch_error_marks_failed_and_records_breaker(
    queue: SqliteQueue, breaker: CircuitBreaker
) -> None:
    src = _FakeSource(fetch_raises=RuntimeError("503 Service Unavailable"))
    queue.enqueue("fake:one", "fake")
    result = run_once(_ctx(queue, breaker, {"fake": src}))
    assert result.outcome == JobOutcome.FAILED
    # First failure of three: job back to pending with error recorded.
    pending = queue.list_pending()
    assert len(pending) == 1
    assert "503" in pending[0].error
    # Breaker counter incremented (one more failure trips it).
    breaker.record_failure("fake")
    assert breaker.allow("fake") is False


def test_run_once_terminal_failure_after_max_attempts(
    queue: SqliteQueue, tmp_path: Path
) -> None:
    # High-threshold breaker so it doesn't intercept before max_attempts trips.
    breaker = CircuitBreaker(tmp_path / "q.db", failure_threshold=99, open_seconds=60)
    try:
        src = _FakeSource(fetch_raises=RuntimeError("permanent"))
        queue.enqueue("fake:one", "fake")
        ctx = WorkerContext(
            queue=queue,
            breaker=breaker,
            sources={"fake": src},
            ingest_ctx=IngestContext(),
            max_attempts=3,
        )
        for _ in range(3):
            run_once(ctx)
        # Job exhausted attempts: no longer pending, no longer leasable.
        assert queue.list_pending() == []
        result = run_once(ctx)
        assert result.outcome == JobOutcome.IDLE
    finally:
        breaker.close()


def test_run_once_success_records_breaker_recovery(
    queue: SqliteQueue, breaker: CircuitBreaker
) -> None:
    breaker.record_failure("fake")
    src = _FakeSource()
    queue.enqueue("fake:one", "fake")
    run_once(_ctx(queue, breaker, {"fake": src}))
    # Success after a failure should reset the breaker.
    for _ in range(10):
        # If reset worked, we need full threshold (2) more failures to open.
        pass
    breaker.record_failure("fake")
    assert breaker.allow("fake") is True
