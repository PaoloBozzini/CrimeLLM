"""Per-source circuit breaker, persisted in SQLite alongside the queue.

Three states: ``closed`` (normal), ``open`` (block until cooldown elapses),
``half_open`` (allow exactly one trial; success → closed, failure → re-open).

State persists across worker restarts because the typical failure mode is
"third-party API is having a bad day for an hour" — losing state on restart
means a flapping worker would hammer the source instead of backing off.

Sharing the SQLite file with ``SqliteQueue`` keeps the on-disk surface to
one file (easy to inspect, easy to nuke for a clean dev reset).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS autofetch_circuit (
  source           TEXT PRIMARY KEY,
  state            TEXT NOT NULL,
  failures         INTEGER NOT NULL DEFAULT 0,
  opened_at        TEXT,
  next_attempt_at  TEXT
);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


class CircuitBreaker:
    """Source-scoped breaker with persisted state."""

    def __init__(
        self,
        path: Path | str,
        *,
        failure_threshold: int = 5,
        open_seconds: int = 3600,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)
        self._threshold = failure_threshold
        self._open_seconds = open_seconds

    # --- API ---------------------------------------------------------------

    def allow(self, source: str) -> bool:
        """Return ``True`` when a fetch attempt should proceed.

        Side effect: transitions ``open`` → ``half_open`` when the cooldown
        expires, so the next call observes the trial-allowed state directly.
        """
        row = self._read(source)
        if row is None or row["state"] == "closed":
            return True
        if row["state"] == "half_open":
            return True
        # state == 'open'
        next_attempt = _parse_iso(row["next_attempt_at"])
        if next_attempt is not None and _now() >= next_attempt:
            self._write(source, state="half_open", failures=row["failures"])
            return True
        return False

    def record_success(self, source: str) -> None:
        """Reset the breaker to ``closed`` with zero failures."""
        self._write(source, state="closed", failures=0)

    def record_failure(self, source: str) -> None:
        """Bump the failure counter; open the breaker once the threshold trips."""
        row = self._read(source)
        prev_state = row["state"] if row else "closed"
        prev_failures = row["failures"] if row else 0
        # half_open + failure → straight back to open (count this as 'enough'
        # signal to keep the cooldown going; don't require N more failures).
        if prev_state == "half_open":
            self._open(source, failures=max(prev_failures, self._threshold))
            return
        new_failures = prev_failures + 1
        if new_failures >= self._threshold:
            self._open(source, failures=new_failures)
        else:
            self._write(source, state="closed", failures=new_failures)

    def close(self) -> None:
        self._conn.close()

    # --- internals ---------------------------------------------------------

    def _open(self, source: str, *, failures: int) -> None:
        opened = _now()
        nxt = opened + timedelta(seconds=self._open_seconds)
        self._conn.execute(
            "INSERT INTO autofetch_circuit (source, state, failures, opened_at, next_attempt_at) "
            "VALUES (?, 'open', ?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET state='open', failures=excluded.failures, "
            "  opened_at=excluded.opened_at, next_attempt_at=excluded.next_attempt_at",
            (source, failures, _iso(opened), _iso(nxt)),
        )

    def _write(self, source: str, *, state: str, failures: int) -> None:
        self._conn.execute(
            "INSERT INTO autofetch_circuit (source, state, failures, opened_at, next_attempt_at) "
            "VALUES (?, ?, ?, NULL, NULL) "
            "ON CONFLICT(source) DO UPDATE SET state=excluded.state, failures=excluded.failures, "
            "  opened_at=NULL, next_attempt_at=NULL",
            (source, state, failures),
        )

    def _read(self, source: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT source, state, failures, opened_at, next_attempt_at "
            "  FROM autofetch_circuit WHERE source = ?",
            (source,),
        ).fetchone()

    # --- test hooks --------------------------------------------------------

    def _set_next_attempt_for_test(self, source: str, when: datetime) -> None:
        self._conn.execute(
            "UPDATE autofetch_circuit SET next_attempt_at = ? WHERE source = ?",
            (_iso(when), source),
        )
