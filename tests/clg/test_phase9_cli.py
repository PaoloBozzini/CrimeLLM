"""Phase 9: CLI surface verification.

Covers:
* All five jurisdictions surface in subcommand `--help` output (no stale
  ``US|EW|UK`` strings).
* `clg query --lang` override threads through to the Answer.
* Answer.to_dict carries `jurisdiction` / `language` / `as_of` so JSON
  consumers can audit.
* Removability matrix smoke: each ingest CLI verb refuses when its
  jurisdiction isn't in `ENABLED_JURISDICTIONS`.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from crimellm.clg.cli import app

runner = CliRunner()


# --- T9.1: help-text refresh ---------------------------------------------


def test_query_help_lists_all_five_jurisdictions():
    r = runner.invoke(app, ["query", "--help"])
    assert r.exit_code == 0
    assert "US|EW|UK|EU|DK" in r.stdout


def test_query_help_mentions_lang_override():
    r = runner.invoke(app, ["query", "--help"])
    assert r.exit_code == 0
    assert "--lang" in r.stdout
    assert "en|da" in r.stdout


def test_graph_search_help_refreshed():
    r = runner.invoke(app, ["graph", "search", "--help"])
    assert r.exit_code == 0
    assert "US|EW|UK|EU|DK" in r.stdout


def test_link_treatment_help_refreshed():
    r = runner.invoke(app, ["link", "treatment", "--help"])
    assert r.exit_code == 0
    assert "US|EW|UK|EU|DK" in r.stdout


def test_embed_help_refreshed():
    r = runner.invoke(app, ["embed", "--help"])
    assert r.exit_code == 0
    assert "US|EW|UK|EU|DK" in r.stdout


# --- T9.2: --lang override end-to-end via FakeSynthesizer ----------------


def test_query_lang_override_forces_da_synthesis(monkeypatch):
    """EN query body + ``--lang da`` → DA disclaimer in the answer.

    Uses Fake synthesizer + a stub run_query to avoid touching Neo4j.
    """
    from datetime import date

    from crimellm.clg import cli as cli_mod
    from crimellm.clg.retrieval import query as query_mod
    from crimellm.clg.retrieval.parse_query import parse_query
    from crimellm.clg.retrieval.synthesize import FakeSynthesizer
    from crimellm.clg.retrieval.seed import Candidate

    captured: dict[str, object] = {}

    def _fake_run_query(question, **kwargs):
        # Capture the kwargs so we can assert language was threaded.
        captured["kwargs"] = dict(kwargs)
        q = parse_query(question).with_overrides(
            language=kwargs.get("language"),
            jurisdiction=kwargs.get("jurisdiction"),
            as_of=kwargs.get("as_of"),
        )
        c = Candidate(
            chunk_id="c1",
            text="Test text body for fake.",
            parent_type="Provision",
            parent_id="eu/celex/32016R0679/article/art.6",
            parent_name="GDPR",
            parent_jurisdiction="EU",
            section_path="art.6",
            version_id="en",
            decision_date=None,
            source="seed",
            base_score=1.0,
            score=1.0,
        )
        return FakeSynthesizer().synthesise(query=q, candidates=[c], good_law={})

    monkeypatch.setattr(cli_mod, "run_query", _fake_run_query, raising=False)
    # The CLI imports run_query lazily inside query_cmd; patch the
    # source module too so the inner import resolves to our stub.
    monkeypatch.setattr(query_mod, "run_query", _fake_run_query, raising=False)
    monkeypatch.setattr(
        "crimellm.clg.retrieval.run_query", _fake_run_query, raising=False
    )

    r = runner.invoke(
        app,
        [
            "query",
            "What does Article 6 GDPR say?",
            "--lang",
            "da",
            "--synth",
            "fake",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert captured["kwargs"]["language"] == "da"

    body = json.loads(r.stdout)
    assert body["language"] == "da"
    # DA disclaimer is in the text body.
    assert "Dette er forskningsstøtte" in body["text"]


# --- Answer.to_dict shape (T9.2) -----------------------------------------


def test_answer_to_dict_carries_resolved_metadata():
    from datetime import date

    from crimellm.clg.retrieval.parse_query import parse_query
    from crimellm.clg.retrieval.seed import Candidate
    from crimellm.clg.retrieval.synthesize import FakeSynthesizer

    q = parse_query("Hvad indebærer straffelovens § 279?").with_overrides(
        as_of=date(2024, 6, 1)
    )
    cand = Candidate(
        chunk_id="c1",
        text="Body.",
        parent_type="Provision",
        parent_id="dk/lbk/2018/502/section/§6/stk.1",
        parent_name="Databeskyttelsesloven",
        parent_jurisdiction="DK",
        section_path="§ 6 stk. 1",
        version_id=None,
        decision_date=None,
        source="seed",
        base_score=1.0,
        score=1.0,
    )
    ans = FakeSynthesizer().synthesise(query=q, candidates=[cand], good_law={})
    d = ans.to_dict()
    assert d["jurisdiction"] == "DK"
    assert d["language"] == "da"
    assert d["as_of"] == "2024-06-01"


# --- T9.3: removability matrix at the CLI surface ------------------------


def test_eurlex_ingest_blocked_when_eu_disabled(monkeypatch):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "US,UK")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    r = runner.invoke(app, ["ingest", "eurlex", "--celex", "32016R0679"])
    assert r.exit_code != 0
    out = r.stderr + r.output
    assert "EU" in out and "enabled_jurisdictions" in out


def test_retsinformation_ingest_blocked_when_dk_disabled(monkeypatch):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "US,EU")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    r = runner.invoke(
        app, ["ingest", "retsinformation", "--items", "lbk/2018/502"]
    )
    assert r.exit_code != 0
    out = r.stderr + r.output
    assert "DK" in out and "enabled_jurisdictions" in out


def test_domstol_ingest_blocked_when_dk_disabled(monkeypatch):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "EU")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    r = runner.invoke(
        app,
        [
            "ingest",
            "domstol",
            "--items",
            "ECLI:DK:HR:2023:1|https://example.org/x.pdf",
        ],
    )
    assert r.exit_code != 0
    out = r.stderr + r.output
    assert "DK" in out and "enabled_jurisdictions" in out


def test_karnov_gate_message_without_key(monkeypatch):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK")
    monkeypatch.delenv("KARNOV_API_KEY", raising=False)
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    r = runner.invoke(app, ["ingest", "karnov"])
    assert r.exit_code == 2
    assert "KARNOV_API_KEY" in r.stdout


def test_ingest_subcommands_list_dk_eu_verbs():
    r = runner.invoke(app, ["ingest", "--help"])
    assert r.exit_code == 0
    for sub in ("eurlex", "retsinformation", "domstol", "karnov"):
        assert sub in r.stdout, f"missing ingest verb `{sub}`"


def test_parse_subcommands_list_dk_eu_verbs():
    r = runner.invoke(app, ["parse", "--help"])
    assert r.exit_code == 0
    for sub in ("eurlex", "retsinformation", "domstol"):
        assert sub in r.stdout, f"missing parse verb `{sub}`"


def test_load_subcommands_list_dk_eu_verbs():
    r = runner.invoke(app, ["load", "--help"])
    assert r.exit_code == 0
    for sub in ("eurlex", "retsinformation", "domstol"):
        assert sub in r.stdout, f"missing load verb `{sub}`"


# --- fixture: reset settings cache around env-based tests ----------------


import pytest


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    yield
    _config.get_settings.cache_clear()
