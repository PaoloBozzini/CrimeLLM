"""Phase H: hardening — concurrency, cite-flood, source-down recovery.

These tests exercise the autofetch subsystem at the failure-mode boundaries
without touching real APIs or Neo4j. Live integration (H.4) lives in the
manual smoke checklist; the unit-level coverage here is what runs in CI.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from crimellm.clg.autofetch.cascade import cascade_from_paths
from crimellm.clg.autofetch.circuit_breaker import CircuitBreaker
from crimellm.clg.autofetch.queue import SqliteQueue
from crimellm.clg.autofetch.resolver import register_rule
from crimellm.clg.autofetch.worker import JobOutcome, WorkerContext, run_once
from crimellm.clg.ingest._base import IngestContext, LoadReport, Source


class _CountingSource(Source):
    name = "ct"

    def __init__(self, *, fail_until: int = 0) -> None:
        self._fail_until = fail_until
        self.attempts: dict[str, int] = {}
        self._lock = threading.Lock()

    def download(self, ctx: IngestContext) -> dict[str, Path]:  # pragma: no cover
        return {}

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:  # pragma: no cover
        yield from ()

    def load(self, ctx: IngestContext) -> LoadReport:
        return LoadReport(source=self.name)

    def supports_single_fetch(self) -> bool:
        return True

    def fetch_one(self, ctx: IngestContext, cite_id: str) -> dict[str, Path]:
        with self._lock:
            self.attempts[cite_id] = self.attempts.get(cite_id, 0) + 1
            n = self.attempts[cite_id]
        if n <= self._fail_until:
            raise RuntimeError("503 simulated")
        return {cite_id: Path(f"/tmp/{cite_id}")}


@pytest.fixture(autouse=True)
def _register_ct_rule():
    register_rule(r"^ct:[a-z0-9]+$", "ct")
    yield
    from crimellm.clg.autofetch import resolver as R

    R._RULES.pop()


# --- H.1: concurrent workers do not double-fetch --------------------------


def test_two_workers_never_double_fetch(tmp_path: Path) -> None:
    queue = SqliteQueue(tmp_path / "q.db")
    breaker = CircuitBreaker(tmp_path / "q.db", failure_threshold=99, open_seconds=60)
    try:
        for i in range(40):
            queue.enqueue(f"ct:job{i}", "ct")

        src = _CountingSource()

        def drain(n_jobs: int) -> None:
            ctx = WorkerContext(
                queue=queue,
                breaker=breaker,
                sources={"ct": src},
                ingest_ctx=IngestContext(),
                max_attempts=3,
            )
            for _ in range(n_jobs):
                run_once(ctx)

        t1 = threading.Thread(target=drain, args=(40,))
        t2 = threading.Thread(target=drain, args=(40,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Every cite fetched exactly once → no double-lease.
        assert all(v == 1 for v in src.attempts.values())
        assert len(src.attempts) == 40
        assert queue.list_pending() == []
    finally:
        queue.close()
        breaker.close()


# --- H.2: adversarial cite-flood -------------------------------------------


def test_cite_flood_unrecognised_shapes_does_not_queue(tmp_path: Path) -> None:
    """1000 garbage cite ids → resolver rejects → cascade enqueues none."""
    queue = SqliteQueue(tmp_path / "q.db")
    try:
        doc = tmp_path / "evil.xml"
        doc.write_text(
            " ".join(f"totally-not-a-cite-{i}" for i in range(1000)),
            encoding="utf-8",
        )
        new = cascade_from_paths(
            [doc], parent_depth=0, max_depth=5, queue=queue
        )
        assert new == []
        assert queue.list_pending() == []
    finally:
        queue.close()


def test_cite_flood_real_shapes_dedups(tmp_path: Path) -> None:
    """100 repeats of one CELEX → exactly one queue row (PK collision)."""
    queue = SqliteQueue(tmp_path / "q.db")
    try:
        doc = tmp_path / "spam.xml"
        doc.write_text(" ".join(["32016R0679"] * 100), encoding="utf-8")
        new = cascade_from_paths(
            [doc], parent_depth=0, max_depth=2, queue=queue
        )
        assert new == ["32016R0679"]
        assert len(queue.list_pending()) == 1
    finally:
        queue.close()


# --- H.3: source-down chaos ------------------------------------------------


def test_source_down_then_recovers(tmp_path: Path) -> None:
    """Repeated 5xx trips breaker; after cooldown the source recovers."""
    queue = SqliteQueue(tmp_path / "q.db")
    breaker = CircuitBreaker(tmp_path / "q.db", failure_threshold=2, open_seconds=60)
    try:
        # Stage 1: source always fails → breaker should trip.
        src = _CountingSource(fail_until=99)
        queue.enqueue("ct:a", "ct")
        queue.enqueue("ct:b", "ct")
        ctx = WorkerContext(
            queue=queue,
            breaker=breaker,
            sources={"ct": src},
            ingest_ctx=IngestContext(),
            max_attempts=99,  # don't terminate jobs during the chaos window
        )

        # Drain until breaker opens (≤4 calls suffice with threshold=2).
        outcomes = [run_once(ctx).outcome for _ in range(4)]
        assert JobOutcome.FAILED in outcomes
        assert JobOutcome.CIRCUIT_OPEN in outcomes
        assert not breaker.allow("ct")

        # Stage 2: source recovers; cool-down elapses; retries succeed.
        src._fail_until = 0
        breaker._set_next_attempt_for_test(
            "ct", datetime.now(timezone.utc) - timedelta(seconds=1)
        )

        for _ in range(20):
            r = run_once(ctx)
            if r.outcome == JobOutcome.IDLE:
                break
        assert queue.list_pending() == []
    finally:
        queue.close()
        breaker.close()
