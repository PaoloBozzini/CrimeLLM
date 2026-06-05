from __future__ import annotations

from typer.testing import CliRunner

from crimellm.clg.cli import app

runner = CliRunner()


def test_help_lists_subcommands() -> None:
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    out = r.stdout
    for sub in ("graph", "ingest", "parse", "link", "embed", "query", "eval"):
        assert sub in out, f"missing subcommand `{sub}` in --help output"


def test_graph_subcommands_listed() -> None:
    r = runner.invoke(app, ["graph", "--help"])
    assert r.exit_code == 0
    for sub in ("init", "status", "wipe", "drop-schema"):
        assert sub in r.stdout


def test_ingest_subcommands_listed() -> None:
    r = runner.invoke(app, ["ingest", "--help"])
    assert r.exit_code == 0
    for sub in ("courtlistener", "uscode", "legislation-uk", "find-case-law"):
        assert sub in r.stdout


def test_wipe_refuses_without_yes() -> None:
    r = runner.invoke(app, ["graph", "wipe"])
    assert r.exit_code == 2
    assert "Refusing" in r.stdout


def test_find_case_law_refuses_without_licence(monkeypatch) -> None:
    monkeypatch.setenv("TNA_COMPUTATIONAL_LICENCE_ACCEPTED", "0")
    from crimellm.clg import config as cfg
    cfg.get_settings.cache_clear()
    r = runner.invoke(app, ["ingest", "find-case-law"])
    assert r.exit_code == 2
    assert "computational-analysis licence" in r.stdout
