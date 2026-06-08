"""Phase A.1: autofetch settings parse correctly with defaults + env overrides.

The autofetch subsystem is gated behind a flag (default off). Tests pin the
default surface so a regression that flips the flag, drops a knob, or changes
the source-QPS map can't ship silently.
"""

from __future__ import annotations

from pathlib import Path

from crimellm.clg.config import Settings


def test_autofetch_disabled_by_default() -> None:
    s = Settings(_env_file=None)
    assert s.autofetch_enabled is False


def test_autofetch_default_queue_path() -> None:
    s = Settings(_env_file=None)
    assert s.autofetch_queue_path == Path("data/autofetch.db")


def test_autofetch_default_limits() -> None:
    s = Settings(_env_file=None)
    assert s.autofetch_max_depth == 2
    assert s.autofetch_max_attempts == 3
    assert s.autofetch_circuit_open_seconds == 3600


def test_autofetch_default_source_qps_has_known_sources() -> None:
    s = Settings(_env_file=None)
    assert "eurlex" in s.autofetch_source_qps
    assert "retsinformation" in s.autofetch_source_qps
    assert "courtlistener" in s.autofetch_source_qps
    assert s.autofetch_source_qps["eurlex"] > 0
    assert s.autofetch_source_qps["eurlex"] <= s.autofetch_source_qps["courtlistener"]


def test_autofetch_env_enable(monkeypatch) -> None:
    monkeypatch.setenv("AUTOFETCH_ENABLED", "true")
    s = Settings(_env_file=None)
    assert s.autofetch_enabled is True


def test_autofetch_env_queue_path(monkeypatch) -> None:
    monkeypatch.setenv("AUTOFETCH_QUEUE_PATH", "/tmp/test-autofetch.db")
    s = Settings(_env_file=None)
    assert s.autofetch_queue_path == Path("/tmp/test-autofetch.db")


def test_autofetch_env_max_depth(monkeypatch) -> None:
    monkeypatch.setenv("AUTOFETCH_MAX_DEPTH", "5")
    s = Settings(_env_file=None)
    assert s.autofetch_max_depth == 5
