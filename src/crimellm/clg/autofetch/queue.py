"""SQLite-backed work queue for the autofetch reconciliation worker.

One backend, deliberately. The queue must survive Neo4j rebuilds, eval
freezes, and Neo4j outages — coupling it to the same store would defeat the
point. WAL mode handles multi-worker concurrency without extra ceremony.

Schema and locking strategy: see ``docs/self-management-autofetch.local.md``
§3. Atomic lease uses ``UPDATE … WHERE cite_id = (SELECT … LIMIT 1)
RETURNING …`` so two workers never claim the same row. Stale leases (worker
crashed mid-job) are reclaimed by the same SELECT predicate — no separate
janitor needed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS autofetch_queue (
  cite_id     TEXT PRIMARY KEY,
  source      TEXT NOT NULL,
  status      TEXT NOT NULL,
  attempts    INTEGER NOT NULL DEFAULT 0,
  depth       INTEGER NOT NULL DEFAULT 0,
  first_seen  TEXT NOT NULL,
  last_tried  TEXT,
  error       TEXT,
  lease_until TEXT
);
CREATE INDEX IF NOT EXISTS idx_status_tried ON autofetch_queue(status, last_tried);
CREATE INDEX IF NOT EXISTS idx_source_status ON autofetch_queue(source, status);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class LeasedJob:
    cite_id: str
    source: str
    depth: int
    attempts: int


@dataclass(frozen=True, slots=True)
class QueueRow:
    cite_id: str
    source: str
    status: str
    attempts: int
    depth: int
    first_seen: str
    last_tried: str | None
    error: str | None
    lease_until: str | None


class SqliteQueue:
    """Idempotent, leaseable work queue."""

    def __init__(self, path: Path | str, *, lease_seconds: int = 300) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` because the worker may hand the conn to
        # short-lived helper threads; SQLite serialises writes anyway under WAL.
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)
        self._lease_seconds = lease_seconds

    # --- mutations ---------------------------------------------------------

    def enqueue(self, cite_id: str, source: str, *, depth: int = 0) -> bool:
        """Insert if not present. Return ``True`` when a new row was created."""
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO autofetch_queue "
            "(cite_id, source, status, depth, first_seen) "
            "VALUES (?, ?, 'pending', ?, ?)",
            (cite_id, source, depth, _now_iso()),
        )
        return cur.rowcount == 1

    def lease(self) -> LeasedJob | None:
        """Atomically claim the oldest pending or stale-leased job.

        The same predicate handles both "new pending" and "previously leased
        by a now-dead worker" — no separate reclaim step needed before each
        lease call, and concurrent workers cannot double-claim.
        """
        now = _now_iso()
        # SQLite's UPDATE…RETURNING (3.35+, shipped with Python 3.11 stdlib)
        # gives us the atomic claim. The subselect picks the oldest eligible
        # row; the WHERE on the outer UPDATE re-checks under the row lock.
        sql = """
        UPDATE autofetch_queue
           SET status      = 'leased',
               lease_until = ?,
               last_tried  = ?,
               attempts    = attempts + 1
         WHERE cite_id = (
           SELECT cite_id FROM autofetch_queue
            WHERE status = 'pending'
               OR (status = 'leased' AND lease_until < ?)
            ORDER BY first_seen ASC
            LIMIT 1
         )
        RETURNING cite_id, source, depth, attempts
        """
        lease_until = _iso(
            datetime.now(timezone.utc).replace(microsecond=0)
            + _timedelta_seconds(self._lease_seconds)
        )
        cur = self._conn.execute(sql, (lease_until, now, now))
        row = cur.fetchone()
        if row is None:
            return None
        return LeasedJob(
            cite_id=row["cite_id"],
            source=row["source"],
            depth=row["depth"],
            attempts=row["attempts"],
        )

    def mark_done(self, cite_id: str) -> None:
        self._conn.execute(
            "UPDATE autofetch_queue SET status = 'done', "
            "lease_until = NULL, error = NULL WHERE cite_id = ?",
            (cite_id,),
        )

    def mark_failed(self, cite_id: str, error: str, *, max_attempts: int = 3) -> None:
        """Record failure; flip to terminal ``failed`` once attempts exhausted.

        Reads current attempts under the same statement so concurrent leases
        don't race the threshold check.
        """
        self._conn.execute(
            "UPDATE autofetch_queue "
            "   SET status = CASE WHEN attempts >= ? THEN 'failed' ELSE 'pending' END, "
            "       error = ?, "
            "       lease_until = NULL "
            " WHERE cite_id = ?",
            (max_attempts, error, cite_id),
        )

    def release(self, cite_id: str) -> None:
        """Return a leased job to ``pending`` without burning its attempt.

        Used when the worker can't process the job through no fault of its
        own (circuit breaker open, source temporarily unconfigured). The
        prior ``lease`` bumped ``attempts``; we roll it back so a backoff
        loop doesn't terminally fail a healthy cite over an outage.
        """
        self._conn.execute(
            "UPDATE autofetch_queue "
            "   SET status = 'pending', lease_until = NULL, "
            "       attempts = MAX(0, attempts - 1) "
            " WHERE cite_id = ?",
            (cite_id,),
        )

    def mark_skipped(self, cite_id: str, reason: str) -> None:
        """Terminal: cite has no resolver mapping. Persist so re-enqueue is cheap."""
        self._conn.execute(
            "UPDATE autofetch_queue SET status = 'skipped', "
            "  lease_until = NULL, error = ? WHERE cite_id = ?",
            (reason, cite_id),
        )

    def reclaim_stale_leases(self) -> int:
        """Flip ``leased`` rows whose lease expired back to ``pending``."""
        now = _now_iso()
        cur = self._conn.execute(
            "UPDATE autofetch_queue SET status = 'pending', lease_until = NULL "
            "WHERE status = 'leased' AND lease_until < ?",
            (now,),
        )
        return cur.rowcount

    # --- reads -------------------------------------------------------------

    def list_pending(self, *, limit: int = 50) -> list[QueueRow]:
        rows = self._conn.execute(
            "SELECT cite_id, source, status, attempts, depth, first_seen, "
            "       last_tried, error, lease_until "
            "  FROM autofetch_queue WHERE status = 'pending' "
            "  ORDER BY first_seen ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [QueueRow(**dict(r)) for r in rows]

    # --- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    # --- test hooks --------------------------------------------------------
    #
    # Tests need to fast-forward lease expiry without sleeping. Exposing this
    # as a clearly-named ``_..._for_test`` helper is cleaner than letting
    # tests reach into the raw connection (which they could anyway, but the
    # named seam makes intent obvious in failure traces).

    def _set_lease_until_for_test(self, cite_id: str, when: datetime) -> None:
        self._conn.execute(
            "UPDATE autofetch_queue SET lease_until = ? WHERE cite_id = ?",
            (_iso(when), cite_id),
        )


def _timedelta_seconds(seconds: int):
    # Imported here to keep the module's import surface tiny.
    from datetime import timedelta

    return timedelta(seconds=seconds)
