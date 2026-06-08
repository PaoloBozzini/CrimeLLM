"""Phase C.0: ``UnsupportedCite`` short-circuits a job to terminal SKIPPED.

A source can raise this when it understands the resolver mapping but
genuinely cannot fetch the cite (slug-shape ELI, gated subscription, etc.).
Treating that as ``FAILED`` would burn attempts pointlessly. Treating it as
``SKIPPED`` makes the queue persist the cite + reason so an operator can
spot it via ``clg autofetch list-pending`` and either promote or wire the
missing path.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from crimellm.clg.autofetch.circuit_breaker import CircuitBreaker
from crimellm.clg.autofetch.exceptions import UnsupportedCite
from crimellm.clg.autofetch.queue import SqliteQueue
from crimellm.clg.autofetch.resolver import register_rule
from crimellm.clg.autofetch.worker import JobOutcome, WorkerContext, run_once
from crimellm.clg.ingest._base import IngestContext, LoadReport, Source


class _UnsupportedSource(Source):
    name = "unsup"

    def download(self, ctx: IngestContext) -> dict[str, Path]:  # pragma: no cover
        return {}

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:  # pragma: no cover
        yield from ()

    def load(self, ctx: IngestContext) -> LoadReport:  # pragma: no cover
        return LoadReport(source=self.name)

    def supports_single_fetch(self) -> bool:
        return True

    def fetch_one(self, ctx: IngestContext, cite_id: str) -> dict[str, Path]:
        raise UnsupportedCite("slug-shape ELI not resolvable to accn")


@pytest.fixture(autouse=True)
def _register_unsup_rule():
    register_rule(r"^unsup:[a-z0-9-]+$", "unsup")
    yield
    from crimellm.clg.autofetch import resolver as R

    R._RULES.pop()


def test_unsupported_cite_marks_skipped(tmp_path: Path) -> None:
    queue = SqliteQueue(tmp_path / "q.db")
    breaker = CircuitBreaker(tmp_path / "q.db", failure_threshold=2, open_seconds=60)
    try:
        queue.enqueue("unsup:slug-shape", "unsup")
        ctx = WorkerContext(
            queue=queue,
            breaker=breaker,
            sources={"unsup": _UnsupportedSource()},
            ingest_ctx=IngestContext(),
            max_attempts=3,
        )
        result = run_once(ctx)
        assert result.outcome == JobOutcome.SKIPPED
        assert "slug-shape ELI" in result.error
        # Skipped is terminal — not in pending, not leasable.
        assert queue.list_pending() == []
        assert run_once(ctx).outcome == JobOutcome.IDLE
    finally:
        queue.close()
        breaker.close()


def test_unsupported_cite_does_not_trip_breaker(tmp_path: Path) -> None:
    queue = SqliteQueue(tmp_path / "q.db")
    breaker = CircuitBreaker(tmp_path / "q.db", failure_threshold=2, open_seconds=60)
    try:
        queue.enqueue("unsup:a", "unsup")
        queue.enqueue("unsup:b", "unsup")
        ctx = WorkerContext(
            queue=queue,
            breaker=breaker,
            sources={"unsup": _UnsupportedSource()},
            ingest_ctx=IngestContext(),
            max_attempts=3,
        )
        run_once(ctx)
        run_once(ctx)
        # Two skips would otherwise hit the failure_threshold=2 — but they
        # are not failures of the source, so the breaker stays closed.
        assert breaker.allow("unsup") is True
    finally:
        queue.close()
        breaker.close()
