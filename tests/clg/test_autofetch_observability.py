"""Phase G: structured logs + breaker visibility in CLI status.

Logs use stdlib ``logging`` so operators wire their preferred formatter
(JSON / plain) via ``logging.basicConfig``. The worker logs one record per
job with all the fields you'd want in a dashboard query.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from crimellm.clg.autofetch.circuit_breaker import CircuitBreaker
from crimellm.clg.autofetch.queue import SqliteQueue
from crimellm.clg.autofetch.resolver import register_rule
from crimellm.clg.autofetch.worker import WorkerContext, run_once
from crimellm.clg.cli.autofetch import app
from crimellm.clg.ingest._base import IngestContext, LoadReport, Source


class _OkSource(Source):
    name = "ok"

    def download(self, ctx: IngestContext) -> dict[str, Path]:  # pragma: no cover
        return {}

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:  # pragma: no cover
        yield from ()

    def load(self, ctx: IngestContext) -> LoadReport:
        return LoadReport(source=self.name)

    def supports_single_fetch(self) -> bool:
        return True

    def fetch_one(self, ctx: IngestContext, cite_id: str) -> dict[str, Path]:
        return {cite_id: Path("/tmp/ok")}


@pytest.fixture(autouse=True)
def _register_ok_rule():
    register_rule(r"^ok:[a-z0-9]+$", "ok")
    yield
    from crimellm.clg.autofetch import resolver as R

    R._RULES.pop()


# --- G.1: structured logs --------------------------------------------------


def test_worker_emits_structured_log_on_success(tmp_path: Path, caplog) -> None:
    queue = SqliteQueue(tmp_path / "q.db")
    breaker = CircuitBreaker(tmp_path / "q.db", failure_threshold=99, open_seconds=60)
    try:
        queue.enqueue("ok:one", "ok", depth=1)
        ctx = WorkerContext(
            queue=queue,
            breaker=breaker,
            sources={"ok": _OkSource()},
            ingest_ctx=IngestContext(),
            max_attempts=3,
        )
        with caplog.at_level(logging.INFO, logger="crimellm.clg.autofetch.worker"):
            run_once(ctx)
        records = [r for r in caplog.records if r.name == "crimellm.clg.autofetch.worker"]
        assert records, "expected at least one worker log record"
        # Each record carries a structured ``extra={...}`` payload with the
        # fields a dashboard query needs.
        rec = records[-1]
        extra = getattr(rec, "job", None)
        assert extra is not None, "worker log must attach extras under 'job'"
        assert extra["cite_id"] == "ok:one"
        assert extra["source"] == "ok"
        assert extra["outcome"] == "ok"
        assert extra["depth"] == 1
        assert extra["attempts"] == 1
        assert "duration_ms" in extra
    finally:
        queue.close()
        breaker.close()


# --- G.2: status surfaces breaker state ------------------------------------


def test_status_json_includes_breaker_state(tmp_path: Path) -> None:
    queue_path = tmp_path / "q.db"
    queue = SqliteQueue(queue_path)
    breaker = CircuitBreaker(queue_path, failure_threshold=2, open_seconds=60)
    try:
        # Trip the breaker for one source.
        breaker.record_failure("eurlex")
        breaker.record_failure("eurlex")
        queue.enqueue("eli/lov/2020/1", "retsinformation")
    finally:
        queue.close()
        breaker.close()

    runner = CliRunner()
    result = runner.invoke(
        app, ["status", "--queue-path", str(queue_path), "--format", "json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "circuits" in payload
    assert payload["circuits"].get("eurlex", {}).get("state") == "open"
