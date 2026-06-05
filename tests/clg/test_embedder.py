"""Embedder ABC + backends + factory."""

from __future__ import annotations

import math

import pytest

from crimellm.clg.embed.embedder import (
    Embedder,
    FakeEmbedder,
    embed_in_batches,
    get_embedder,
)


def test_fake_embedder_dim_matches() -> None:
    emb = FakeEmbedder(dim=32)
    v = emb.embed("hello")
    assert len(v) == 32


def test_fake_embedder_unit_norm() -> None:
    v = FakeEmbedder(dim=64).embed("any text")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-9


def test_fake_embedder_deterministic() -> None:
    emb = FakeEmbedder(dim=32)
    a = emb.embed("legal text")
    b = emb.embed("legal text")
    c = emb.embed("different text")
    assert a == b
    assert a != c


def test_fake_embedder_batch_preserves_order() -> None:
    emb = FakeEmbedder(dim=16)
    out = emb.embed_batch(["one", "two", "three"])
    assert len(out) == 3
    assert out[0] == emb.embed("one")
    assert out[2] == emb.embed("three")


def test_embed_in_batches_handles_remainder() -> None:
    emb = FakeEmbedder(dim=8)
    texts = [f"item-{i}" for i in range(10)]
    out = embed_in_batches(emb, texts, batch_size=3)
    assert len(out) == 10
    assert isinstance(out[0], list) and len(out[0]) == 8


def test_get_embedder_fake_via_explicit_arg() -> None:
    emb = get_embedder("fake")
    assert isinstance(emb, FakeEmbedder)


def test_get_embedder_falls_back_to_fake_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from crimellm.clg import config as cfg

    cfg.get_settings.cache_clear()
    emb = get_embedder()  # auto
    assert isinstance(emb, Embedder)


def test_get_embedder_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown embedder backend"):
        get_embedder("nonsense")


# --- sentence-transformers (skip if extra not installed) -------------------


def _st_installed() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _st_installed(),
    reason="sentence-transformers not installed (needs the [classifier] extra)",
)
def test_sentence_transformer_backend_minilm() -> None:
    from crimellm.clg.embed.embedder import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder()  # default model
    assert emb.dim == 384
    vecs = emb.embed_batch(["hello world", "another text"])
    assert len(vecs) == 2
    assert all(len(v) == 384 for v in vecs)
    # MiniLM with normalize_embeddings=True returns unit vectors.
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert 0.99 < norm < 1.01


@pytest.mark.skipif(not _st_installed(), reason="sentence-transformers not installed")
def test_get_embedder_routes_to_sentence_transformers() -> None:
    from crimellm.clg.embed.embedder import SentenceTransformerEmbedder

    for alias in ("sentence-transformers", "st", "local", "minilm"):
        emb = get_embedder(alias)
        assert isinstance(emb, SentenceTransformerEmbedder)


def test_get_embedder_st_without_install_raises_clear_error(monkeypatch) -> None:
    """If someone asks for st but the package isn't installed, raise a useful error."""
    if _st_installed():
        pytest.skip("sentence-transformers is installed; this checks the missing-extra path")
    with pytest.raises(ImportError, match="sentence-transformers"):
        get_embedder("st")
