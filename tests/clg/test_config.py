from __future__ import annotations

import importlib

from crimellm.clg.config import Settings


def test_defaults_load() -> None:
    s = Settings(_env_file=None)
    assert s.neo4j_uri.startswith("bolt://")
    assert s.neo4j_user == "neo4j"
    assert s.embedding_model == "voyage-law-2"
    assert s.embedding_dim == 1024


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    s = Settings(_env_file=None)
    assert s.neo4j_uri == "bolt://example:7687"
    assert s.embedding_dim == 768


def test_models_importable() -> None:
    mod = importlib.import_module("crimellm.clg.models")
    assert hasattr(mod, "Case")
    assert hasattr(mod, "Provision")
    assert hasattr(mod, "Citation")
