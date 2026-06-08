"""Phase B.1: SqliteQueue contract.

The queue is the autofetch worker's only state store. It must be safe under:
- Re-enqueue of the same cite_id (idempotent PK).
- Concurrent workers (atomic lease via UPDATE…RETURNING).
- Crashed workers (stale leases reclaimed on next ``lease`` call).
- Repeated failures (back to pending up to ``max_attempts``, then ``failed``).

Tests use an on-disk SQLite under ``tmp_path`` so WAL mode is real and the
schema-creation path runs end-to-end (in-memory ``:memory:`` databases don't
honour WAL semantics).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from crimellm.clg.autofetch.queue import LeasedJob, SqliteQueue


@pytest.fixture
def queue(tmp_path: Path) -> SqliteQueue:
    q = SqliteQueue(tmp_path / "q.db", lease_seconds=60)
    yield q
    q.close()


# --- enqueue ---------------------------------------------------------------


def test_enqueue_new_returns_true(queue: SqliteQueue) -> None:
    assert queue.enqueue("ECLI:DK:HR:2020:1", "retsinformation") is True


def test_enqueue_duplicate_returns_false(queue: SqliteQueue) -> None:
    queue.enqueue("ECLI:DK:HR:2020:1", "retsinformation")
    assert queue.enqueue("ECLI:DK:HR:2020:1", "retsinformation") is False


def test_enqueue_records_depth(queue: SqliteQueue) -> None:
    queue.enqueue("ECLI:DK:HR:2020:1", "retsinformation", depth=2)
    pending = queue.list_pending()
    assert pending[0].depth == 2


# --- lease ------------------------------------------------------------------


def test_lease_returns_none_when_empty(queue: SqliteQueue) -> None:
    assert queue.lease() is None


def test_lease_claims_pending_job(queue: SqliteQueue) -> None:
    queue.enqueue("c1", "eurlex")
    job = queue.lease()
    assert job is not None
    assert job.cite_id == "c1"
    assert job.source == "eurlex"
    assert job.attempts == 1


def test_lease_skips_leased_job(queue: SqliteQueue) -> None:
    queue.enqueue("c1", "eurlex")
    queue.enqueue("c2", "eurlex")
    first = queue.lease()
    second = queue.lease()
    assert first is not None
    assert second is not None
    assert first.cite_id != second.cite_id


def test_lease_fifo_order(queue: SqliteQueue) -> None:
    queue.enqueue("c1", "eurlex")
    queue.enqueue("c2", "eurlex")
    queue.enqueue("c3", "eurlex")
    assert queue.lease().cite_id == "c1"
    assert queue.lease().cite_id == "c2"
    assert queue.lease().cite_id == "c3"


# --- mark_done / mark_failed -----------------------------------------------


def test_mark_done_removes_from_pending(queue: SqliteQueue) -> None:
    queue.enqueue("c1", "eurlex")
    job = queue.lease()
    queue.mark_done(job.cite_id)
    assert queue.list_pending() == []


def test_mark_failed_under_max_attempts_returns_to_pending(queue: SqliteQueue) -> None:
    queue.enqueue("c1", "eurlex")
    job = queue.lease()
    queue.mark_failed(job.cite_id, "transient 503", max_attempts=3)
    pending = queue.list_pending()
    assert len(pending) == 1
    assert pending[0].cite_id == "c1"
    assert pending[0].error == "transient 503"


def test_mark_failed_at_max_attempts_marks_failed(queue: SqliteQueue) -> None:
    queue.enqueue("c1", "eurlex")
    # attempt 1
    job = queue.lease()
    queue.mark_failed(job.cite_id, "err", max_attempts=3)
    # attempt 2
    job = queue.lease()
    queue.mark_failed(job.cite_id, "err", max_attempts=3)
    # attempt 3 — should now be failed-terminal
    job = queue.lease()
    queue.mark_failed(job.cite_id, "err", max_attempts=3)
    assert queue.list_pending() == []
    assert queue.lease() is None


# --- reclaim_stale_leases --------------------------------------------------


def test_reclaim_stale_leases_returns_count(queue: SqliteQueue) -> None:
    queue.enqueue("c1", "eurlex")
    queue.enqueue("c2", "eurlex")
    queue.lease()
    queue.lease()
    # Force-expire both leases.
    queue._set_lease_until_for_test("c1", datetime.now(timezone.utc) - timedelta(minutes=10))
    queue._set_lease_until_for_test("c2", datetime.now(timezone.utc) - timedelta(minutes=10))
    assert queue.reclaim_stale_leases() == 2


def test_lease_picks_up_stale_leased_job(queue: SqliteQueue) -> None:
    queue.enqueue("c1", "eurlex")
    first = queue.lease()
    queue._set_lease_until_for_test(
        "c1", datetime.now(timezone.utc) - timedelta(minutes=10)
    )
    second = queue.lease()
    assert second is not None
    assert second.cite_id == "c1"
    assert second.attempts == 2  # attempt counter advances on re-lease


# --- schema / wal ----------------------------------------------------------


def test_wal_mode_active(queue: SqliteQueue) -> None:
    cur = queue._conn.execute("PRAGMA journal_mode")
    mode = cur.fetchone()[0]
    assert mode.lower() == "wal"


def test_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "q.db"
    q1 = SqliteQueue(path)
    q1.enqueue("c1", "eurlex")
    q1.close()
    q2 = SqliteQueue(path)
    try:
        pending = q2.list_pending()
        assert len(pending) == 1
        assert pending[0].cite_id == "c1"
    finally:
        q2.close()
