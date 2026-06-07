"""Phase 2: multilingual embedder default + per-jurisdiction rebuild plumbing.

Covers code paths that don't need a live Neo4j: default model name, CSV
parser, CLI validation gates. The full embed-rebuild round-trip
(DELETE chunks → re-embed → MERGE) is covered by the Phase-2 smoke step
in ``test_embed_load.py`` when neo4j is available.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from crimellm.clg.cli import app
from crimellm.clg.cli._common import parse_jurisdiction_csv
from crimellm.clg.config import KNOWN_MODEL_DIMS, Settings, dim_for_model
from crimellm.clg.embed.embedder import (
    SentenceTransformerEmbedder,
    _is_hf_model_name,
)

runner = CliRunner()


def test_default_embedding_model_is_qwen3_8b():
    # Production default: open-source, multilingual, MTEB top-tier.
    fields = Settings.model_fields
    assert fields["embedding_model"].default == "Qwen/Qwen3-Embedding-8B"
    assert fields["embedding_dim"].default == 4096
    # Fallback recorded too.
    assert fields["embedding_fallback_model"].default == "BAAI/bge-m3"


def test_sentence_transformer_class_default_is_qwen3_8b():
    import inspect

    sig = inspect.signature(SentenceTransformerEmbedder.__init__)
    assert sig.parameters["model"].default == "Qwen/Qwen3-Embedding-8B"


def test_known_model_dims_covers_picks():
    # Production + fallback + dev picks must all have a registered dim so
    # auto-derive can't silently mismatch the vector index.
    assert KNOWN_MODEL_DIMS["Qwen/Qwen3-Embedding-8B"] == 4096
    assert KNOWN_MODEL_DIMS["Qwen/Qwen3-Embedding-0.6B"] == 1024
    assert KNOWN_MODEL_DIMS["BAAI/bge-m3"] == 1024
    assert KNOWN_MODEL_DIMS["sentence-transformers/all-MiniLM-L6-v2"] == 384


def test_dim_for_model_returns_none_for_unknown():
    assert dim_for_model("not-a-real-model") is None


def test_embedding_dim_auto_derives_from_known_model(monkeypatch):
    # Set only EMBEDDING_MODEL — dim must come from the registry.
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    s = Settings(_env_file=None)
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024


def test_embedding_dim_explicit_override_wins(monkeypatch):
    # User-set dim must override the registry derivation.
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    s = Settings(_env_file=None)
    assert s.embedding_dim == 768


def test_hf_routing_detects_qwen_and_bge():
    assert _is_hf_model_name("Qwen/Qwen3-Embedding-8B")
    assert _is_hf_model_name("BAAI/bge-m3")
    assert _is_hf_model_name("sentence-transformers/all-MiniLM-L6-v2")
    assert not _is_hf_model_name("voyage-multilingual-2")
    assert not _is_hf_model_name("text-embedding-3-large")


def test_parse_jurisdiction_csv_basic():
    assert parse_jurisdiction_csv("dk,EU, us") == ["DK", "EU", "US"]


def test_parse_jurisdiction_csv_dedupes():
    assert parse_jurisdiction_csv("DK, dk, DK") == ["DK"]


def test_parse_jurisdiction_csv_drops_empty():
    assert parse_jurisdiction_csv(",,DK,,") == ["DK"]


def test_parse_jurisdiction_csv_empty_string():
    assert parse_jurisdiction_csv("") == []


def test_embed_rebuild_rejects_unknown_jurisdiction(monkeypatch):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK,EU")
    # Bust the lru_cache so the env var takes effect.
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    result = runner.invoke(
        app,
        ["embed-rebuild", "--jurisdiction", "DK,XX", "--yes"],
    )
    assert result.exit_code != 0
    combined = result.stderr + result.output
    assert "XX" in combined and "enabled_jurisdictions" in combined


def test_embed_rebuild_requires_yes(monkeypatch):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK,EU")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    result = runner.invoke(
        app,
        ["embed-rebuild", "--jurisdiction", "DK"],
    )
    assert result.exit_code == 2
    assert "Refusing" in result.stdout or "--yes" in result.stdout


def test_embed_rebuild_empty_csv_rejected(monkeypatch):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK,EU")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    result = runner.invoke(
        app,
        ["embed-rebuild", "--jurisdiction", ",,", "--yes"],
    )
    assert result.exit_code != 0


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    yield
    _config.get_settings.cache_clear()
