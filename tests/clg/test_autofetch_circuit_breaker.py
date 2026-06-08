"""Phase B.2: per-source CircuitBreaker.

States: ``closed`` (allow), ``open`` (block until ``next_attempt_at``),
``half_open`` (allow one trial; success closes, failure re-opens). Persisted
in the same SQLite file as the queue so worker restarts preserve breaker
state — an outage that just opened a breaker shouldn't reset to closed when
the worker re-launches.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from crimellm.clg.autofetch.circuit_breaker import CircuitBreaker


@pytest.fixture
def breaker(tmp_path: Path) -> CircuitBreaker:
    cb = CircuitBreaker(
        tmp_path / "q.db",
        failure_threshold=3,
        open_seconds=60,
    )
    yield cb
    cb.close()


def test_closed_by_default(breaker: CircuitBreaker) -> None:
    assert breaker.allow("eurlex") is True


def test_failures_below_threshold_keep_breaker_closed(breaker: CircuitBreaker) -> None:
    breaker.record_failure("eurlex")
    breaker.record_failure("eurlex")
    assert breaker.allow("eurlex") is True


def test_failures_at_threshold_open_breaker(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure("eurlex")
    assert breaker.allow("eurlex") is False


def test_success_resets_failure_count(breaker: CircuitBreaker) -> None:
    breaker.record_failure("eurlex")
    breaker.record_failure("eurlex")
    breaker.record_success("eurlex")
    breaker.record_failure("eurlex")
    breaker.record_failure("eurlex")
    # Only two failures since last reset — still closed.
    assert breaker.allow("eurlex") is True


def test_open_breaker_recovers_after_window(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure("eurlex")
    assert breaker.allow("eurlex") is False
    # Force the cooldown to expire.
    breaker._set_next_attempt_for_test(
        "eurlex", datetime.now(timezone.utc) - timedelta(seconds=1)
    )
    # First call transitions open → half_open and allows one trial.
    assert breaker.allow("eurlex") is True


def test_half_open_failure_reopens(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure("eurlex")
    breaker._set_next_attempt_for_test(
        "eurlex", datetime.now(timezone.utc) - timedelta(seconds=1)
    )
    breaker.allow("eurlex")  # transitions to half_open
    breaker.record_failure("eurlex")
    assert breaker.allow("eurlex") is False


def test_half_open_success_closes(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure("eurlex")
    breaker._set_next_attempt_for_test(
        "eurlex", datetime.now(timezone.utc) - timedelta(seconds=1)
    )
    breaker.allow("eurlex")  # half_open
    breaker.record_success("eurlex")
    assert breaker.allow("eurlex") is True


def test_state_isolated_per_source(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure("eurlex")
    assert breaker.allow("eurlex") is False
    assert breaker.allow("retsinformation") is True


def test_state_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "q.db"
    cb1 = CircuitBreaker(path, failure_threshold=3, open_seconds=60)
    for _ in range(3):
        cb1.record_failure("eurlex")
    assert cb1.allow("eurlex") is False
    cb1.close()
    cb2 = CircuitBreaker(path, failure_threshold=3, open_seconds=60)
    try:
        assert cb2.allow("eurlex") is False
    finally:
        cb2.close()
