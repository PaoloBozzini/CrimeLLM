"""``crimellm.env`` — .env discovery + load precedence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from crimellm.env import find_dotenv, load_env


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_find_dotenv_finds_in_cwd(isolated_cwd: Path) -> None:
    env = isolated_cwd / ".env"
    env.write_text("FOO=bar\n", encoding="utf-8")
    assert find_dotenv() == env


def test_find_dotenv_walks_up(isolated_cwd: Path) -> None:
    parent_env = isolated_cwd / ".env"
    parent_env.write_text("FOO=parent\n", encoding="utf-8")
    nested = isolated_cwd / "deep" / "deeper"
    nested.mkdir(parents=True)
    os.chdir(nested)
    assert find_dotenv() == parent_env


def test_find_dotenv_returns_none_when_absent(isolated_cwd: Path) -> None:
    assert find_dotenv() is None


def test_load_env_sets_values(isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRIMELLM_TEST_VAR", raising=False)
    (isolated_cwd / ".env").write_text("CRIMELLM_TEST_VAR=hello\n", encoding="utf-8")
    used = load_env()
    assert used is not None
    assert os.environ.get("CRIMELLM_TEST_VAR") == "hello"


def test_load_env_does_not_override_by_default(
    isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRIMELLM_TEST_VAR", "already_set")
    (isolated_cwd / ".env").write_text("CRIMELLM_TEST_VAR=from_file\n", encoding="utf-8")
    load_env()
    assert os.environ.get("CRIMELLM_TEST_VAR") == "already_set"


def test_load_env_override_true_replaces(
    isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRIMELLM_TEST_VAR", "already_set")
    (isolated_cwd / ".env").write_text("CRIMELLM_TEST_VAR=from_file\n", encoding="utf-8")
    load_env(override=True)
    assert os.environ.get("CRIMELLM_TEST_VAR") == "from_file"


def test_load_env_returns_none_when_missing(isolated_cwd: Path) -> None:
    assert load_env() is None
