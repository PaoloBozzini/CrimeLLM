"""Phase B.5: ``clg autofetch ...`` CLI surface.

CLI is a thin shell over ``SqliteQueue``: ``enqueue``, ``status``,
``list-pending``, ``promote``. Tests use a temp DB so we exercise the real
SQLite path (no mocks), but skip ``drain`` here — drain wires a worker and
real ``Source`` instances; B.6 covers it end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from crimellm.clg.autofetch.queue import SqliteQueue
from crimellm.clg.cli.autofetch import app


runner = CliRunner()


@pytest.fixture
def queue_path(tmp_path: Path) -> Path:
    return tmp_path / "autofetch.db"


# --- enqueue ---------------------------------------------------------------


def test_enqueue_new_cite(queue_path: Path) -> None:
    result = runner.invoke(
        app,
        ["enqueue", "eli/lov/2020/171", "--queue-path", str(queue_path)],
    )
    assert result.exit_code == 0
    assert "eli/lov/2020/171" in result.stdout
    q = SqliteQueue(queue_path)
    try:
        pending = q.list_pending()
        assert len(pending) == 1
        assert pending[0].cite_id == "eli/lov/2020/171"
        assert pending[0].source == "retsinformation"
    finally:
        q.close()


def test_enqueue_unknown_cite_id_exits_non_zero(queue_path: Path) -> None:
    result = runner.invoke(
        app, ["enqueue", "totally-unknown", "--queue-path", str(queue_path)]
    )
    assert result.exit_code != 0
    assert "no resolver match" in result.stdout.lower()


def test_enqueue_explicit_source_override(queue_path: Path) -> None:
    """``--source`` lets the operator force a source even if resolver disagrees."""
    result = runner.invoke(
        app,
        [
            "enqueue",
            "U.2010.1234.H",
            "--source",
            "karnov",
            "--queue-path",
            str(queue_path),
        ],
    )
    assert result.exit_code == 0
    q = SqliteQueue(queue_path)
    try:
        assert q.list_pending()[0].source == "karnov"
    finally:
        q.close()


# --- status ----------------------------------------------------------------


def test_status_reports_counts(queue_path: Path) -> None:
    q = SqliteQueue(queue_path)
    try:
        q.enqueue("eli/lov/2020/1", "retsinformation")
        q.enqueue("eli/lov/2020/2", "retsinformation")
        q.enqueue("ECLI:EU:C:2014:317", "eurlex")
    finally:
        q.close()
    result = runner.invoke(
        app, ["status", "--queue-path", str(queue_path), "--format", "json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pending"] == 3
    assert payload["by_source"]["retsinformation"] == 2
    assert payload["by_source"]["eurlex"] == 1


def test_status_empty_queue(queue_path: Path) -> None:
    result = runner.invoke(
        app, ["status", "--queue-path", str(queue_path), "--format", "json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pending"] == 0


# --- list-pending ----------------------------------------------------------


def test_list_pending_outputs_cite_ids(queue_path: Path) -> None:
    q = SqliteQueue(queue_path)
    try:
        q.enqueue("eli/lov/2020/1", "retsinformation")
    finally:
        q.close()
    result = runner.invoke(
        app, ["list-pending", "--queue-path", str(queue_path)]
    )
    assert result.exit_code == 0
    assert "eli/lov/2020/1" in result.stdout


# --- promote ---------------------------------------------------------------
#
# Phase F lands the full validated-flag flip in Neo4j. The CLI surface only
# needs to accept the cite id and exit cleanly; the actual MERGE wires up
# in F.3. For B we pin the command exists + accepts the arg.


def test_promote_command_exists(queue_path: Path) -> None:
    result = runner.invoke(app, ["promote", "--help"])
    assert result.exit_code == 0
    assert "cite_id" in result.stdout.lower() or "cite-id" in result.stdout.lower()
