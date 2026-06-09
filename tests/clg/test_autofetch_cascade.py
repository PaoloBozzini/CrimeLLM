"""Phase E.1+E.2: cascade walker enqueues child cites at depth+1.

After ``Source.fetch_one`` writes a doc to disk, the worker pipes the new
file contents through the cascade. Cite ids inside the doc get enqueued
with ``depth = parent_depth + 1``; anything past
``Settings.autofetch_max_depth`` is dropped to keep the queue bounded.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from crimellm.clg.autofetch.cascade import cascade_from_paths
from crimellm.clg.autofetch.queue import SqliteQueue


@pytest.fixture
def queue(tmp_path: Path) -> SqliteQueue:
    q = SqliteQueue(tmp_path / "q.db")
    yield q
    q.close()


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_cascade_enqueues_celex_found_in_doc(queue: SqliteQueue, tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "doc.xml",
        "<x>This implements Regulation 32016R0679 and 32019L0770.</x>",
    )
    new = cascade_from_paths([p], parent_depth=0, max_depth=2, queue=queue)
    assert set(new) == {"32016R0679", "32019L0770"}
    pending = {row.cite_id: row.depth for row in queue.list_pending()}
    assert pending == {"32016R0679": 1, "32019L0770": 1}


def test_cascade_respects_max_depth(queue: SqliteQueue, tmp_path: Path) -> None:
    p = _write(tmp_path, "doc.xml", "<x>References 32016R0679</x>")
    # parent already at max_depth=2 → child at 3 should NOT enqueue.
    new = cascade_from_paths([p], parent_depth=2, max_depth=2, queue=queue)
    assert new == []
    assert queue.list_pending() == []


def test_cascade_dedups_within_one_call(queue: SqliteQueue, tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "doc.xml",
        "<x>32016R0679 appears, 32016R0679 again, 32016R0679 thrice.</x>",
    )
    new = cascade_from_paths([p], parent_depth=0, max_depth=2, queue=queue)
    assert new == ["32016R0679"]
    assert len(queue.list_pending()) == 1


def test_cascade_skips_cites_already_queued(queue: SqliteQueue, tmp_path: Path) -> None:
    queue.enqueue("32016R0679", "eurlex")
    p = _write(tmp_path, "doc.xml", "<x>32016R0679</x>")
    new = cascade_from_paths([p], parent_depth=0, max_depth=2, queue=queue)
    # Idempotent: already-queued cite isn't re-enqueued. ``new`` only tracks
    # rows actually inserted by this cascade call.
    assert new == []
    rows = queue.list_pending()
    assert len(rows) == 1


def test_cascade_walks_multiple_files(queue: SqliteQueue, tmp_path: Path) -> None:
    a = _write(tmp_path, "a.xml", "<x>32016R0679</x>")
    b = _write(tmp_path, "b.xml", "<x>eli/lov/2020/171</x>")
    new = cascade_from_paths([a, b], parent_depth=0, max_depth=2, queue=queue)
    assert set(new) == {"32016R0679", "eli/lov/2020/171"}


def test_cascade_ignores_binary_garbage(queue: SqliteQueue, tmp_path: Path) -> None:
    p = tmp_path / "garbage.bin"
    p.write_bytes(b"\xff\xfe\x00\x01invalid-utf8")
    # Should not crash; just return [].
    new = cascade_from_paths([p], parent_depth=0, max_depth=2, queue=queue)
    assert new == []


# --- worker integration (E.2) ---------------------------------------------


def test_worker_run_once_cascades_after_success(tmp_path: Path) -> None:
    from collections.abc import Iterator as _Iter

    from crimellm.clg.autofetch.circuit_breaker import CircuitBreaker
    from crimellm.clg.autofetch.resolver import register_rule
    from crimellm.clg.autofetch.worker import (
        JobOutcome,
        WorkerContext,
        run_once,
    )
    from crimellm.clg.ingest._base import IngestContext, LoadReport, Source

    body = b"<x>The doc cites 32016R0679 internally.</x>"

    class _CitingSource(Source):
        name = "citing"

        def download(self, ctx: IngestContext) -> dict[str, Path]:  # pragma: no cover
            return {}

        def parse(self, ctx: IngestContext) -> _Iter[tuple[str, Any]]:  # pragma: no cover
            yield from ()

        def load(self, ctx: IngestContext) -> LoadReport:
            return LoadReport(source=self.name)

        def supports_single_fetch(self) -> bool:
            return True

        def fetch_one(self, ctx: IngestContext, cite_id: str) -> dict[str, Path]:
            out = ctx.source_raw_dir(self.name) / "out.xml"
            out.write_bytes(body)
            return {cite_id: out}

    register_rule(r"^citing:[a-z0-9]+$", "citing")
    try:
        queue = SqliteQueue(tmp_path / "q.db")
        breaker = CircuitBreaker(tmp_path / "q.db", failure_threshold=99, open_seconds=60)
        try:
            queue.enqueue("citing:root", "citing", depth=0)
            ctx = WorkerContext(
                queue=queue,
                breaker=breaker,
                sources={"citing": _CitingSource()},
                ingest_ctx=IngestContext(raw_dir=tmp_path / "raw"),
                max_attempts=3,
                cascade_max_depth=2,
            )
            result = run_once(ctx)
            assert result.outcome == JobOutcome.OK
            # The CELEX inside the doc should now be queued at depth 1.
            pending = {r.cite_id: r.depth for r in queue.list_pending()}
            assert pending == {"32016R0679": 1}
        finally:
            queue.close()
            breaker.close()
    finally:
        from crimellm.clg.autofetch import resolver as R

        R._RULES.pop()
