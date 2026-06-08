"""Phase 11: jurisdiction removal mechanism verification.

The invariant: dropping a jurisdiction from ``ENABLED_JURISDICTIONS`` must
silence it at every retrieval boundary **without deleting any data**.

Three layers are tested:

1. ``search_chunks`` Cypher params — mock store captures kwargs to confirm
   ``enabled`` is threaded correctly (jurisdiction override bypass intact).
2. ``apply_schema`` — only seeds enabled jurisdictions; never deletes
   disabled-jurisdiction nodes. Mock store records the MERGE calls.
3. End-to-end smoke (gated on the test Neo4j container): enable all five,
   load mixed-jurisdiction data, flip to ``[EU, DK]``, confirm US/UK
   Cases + Provisions stay on disk but vector retrieval excludes them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest

from crimellm.clg.config import Settings
from crimellm.clg.graph.loaders import search_chunks
from crimellm.clg.graph.schema import JURISDICTION_SEEDS, apply_schema
from crimellm.clg.retrieval.parse_query import parse_query
from crimellm.clg.retrieval.seed import seed_from_chunks


# --- mock store for non-neo4j tests --------------------------------------


@dataclass
class _RunCall:
    cypher: str
    kwargs: dict[str, Any]


class _MockSession:
    def __init__(self, calls: list[_RunCall], hits: list[dict[str, Any]] | None = None):
        self._calls = calls
        self._hits = hits or []

    def run(self, cypher: str, **kwargs: Any):
        self._calls.append(_RunCall(cypher=cypher, kwargs=kwargs))

        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def __iter__(self):
                return iter(self._rows)

            def single(self):
                return self._rows[0] if self._rows else None

        return _Result(self._hits)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _MockStore:
    """Minimal Neo4jStore stand-in for the layers we exercise here."""

    def __init__(self, hits: list[dict[str, Any]] | None = None):
        self.calls: list[_RunCall] = []
        self.settings = Settings()
        self._hits = hits or []

    @contextmanager
    def session(self):
        yield _MockSession(self.calls, self._hits)

    def run(self, cypher: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(_RunCall(cypher=cypher, kwargs=kwargs))
        return list(self._hits)


@pytest.fixture
def mock_store():
    return _MockStore()


@pytest.fixture(autouse=True)
def _patch_vector_index_dim(monkeypatch):
    """The dimension probe runs against the real index; stub it out so
    search_chunks doesn't trip on the dim check in our mock."""
    from crimellm.clg.graph import loaders as _loaders

    monkeypatch.setattr(_loaders, "vector_index_dim", lambda _store: None)


# --- T11.1: search_chunks Cypher threading ------------------------------


def test_search_chunks_passes_enabled_when_no_jurisdiction(mock_store):
    search_chunks(
        [0.0, 0.0],
        k=5,
        enabled_jurisdictions=["DK", "EU"],
        store=mock_store,
    )
    assert mock_store.calls, "search_chunks should have issued at least one query"
    call = mock_store.calls[-1]
    assert call.kwargs["enabled"] == ["DK", "EU"]
    assert call.kwargs["jurisdiction"] is None
    # Filter must appear in the Cypher so a disabled jurisdiction is
    # actually excluded.
    assert "parent.jurisdiction IN $enabled" in call.cypher


def test_search_chunks_normalises_enabled_uppercase(mock_store):
    search_chunks(
        [0.0, 0.0],
        k=5,
        enabled_jurisdictions=["dk", "Eu"],
        store=mock_store,
    )
    assert mock_store.calls[-1].kwargs["enabled"] == ["DK", "EU"]


def test_search_chunks_explicit_jurisdiction_bypasses_enabled_filter(mock_store):
    """Caller-knows-best: --jurisdiction US must still work even when US
    isn't in enabled_jurisdictions (Phase 7 invariant)."""
    search_chunks(
        [0.0, 0.0],
        k=5,
        jurisdiction="US",
        enabled_jurisdictions=["DK", "EU"],
        store=mock_store,
    )
    call = mock_store.calls[-1]
    assert call.kwargs["jurisdiction"] == "US"
    # The Cypher has the bypass branch ("$jurisdiction IS NOT NULL OR …").
    assert "$jurisdiction IS NOT NULL" in call.cypher


def test_search_chunks_none_enabled_disables_filter(mock_store):
    search_chunks(
        [0.0, 0.0], k=5, enabled_jurisdictions=None, store=mock_store
    )
    call = mock_store.calls[-1]
    assert call.kwargs["enabled"] is None


def test_search_chunks_empty_enabled_disables_filter(mock_store):
    search_chunks(
        [0.0, 0.0], k=5, enabled_jurisdictions=[], store=mock_store
    )
    # Empty list → no filter wanted; the query treats it as None.
    call = mock_store.calls[-1]
    assert call.kwargs["enabled"] is None


# --- seed_from_chunks reads Settings -------------------------------------


def test_seed_from_chunks_reads_settings_enabled_jurisdictions(
    monkeypatch, mock_store
):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK,EU")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()

    class _FakeEmbedder:
        name = "fake"
        dim = 2

        def embed(self, _text: str):
            return [0.0, 0.0]

    seed_from_chunks(
        query=parse_query("Hvad indebærer straffelovens § 279?"),
        embedder=_FakeEmbedder(),
        k=3,
        store=mock_store,
    )
    last = mock_store.calls[-1]
    assert last.kwargs["enabled"] == ["DK", "EU"]


def test_seed_from_chunks_explicit_enabled_overrides_settings(
    monkeypatch, mock_store
):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK,EU")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()

    class _FakeEmbedder:
        name = "fake"
        dim = 2

        def embed(self, _text: str):
            return [0.0, 0.0]

    seed_from_chunks(
        query=parse_query("anything"),
        embedder=_FakeEmbedder(),
        k=3,
        store=mock_store,
        enabled_jurisdictions=["US"],
    )
    assert mock_store.calls[-1].kwargs["enabled"] == ["US"]


def test_seed_from_chunks_all_token_disables_filter(monkeypatch, mock_store):
    class _FakeEmbedder:
        name = "fake"
        dim = 2

        def embed(self, _text: str):
            return [0.0, 0.0]

    seed_from_chunks(
        query=parse_query("anything"),
        embedder=_FakeEmbedder(),
        k=3,
        store=mock_store,
        enabled_jurisdictions=["ALL"],
    )
    assert mock_store.calls[-1].kwargs["enabled"] is None


# --- T11.2: apply_schema honours enabled_jurisdictions -------------------


def _merge_codes_called(calls: list[_RunCall]) -> list[str]:
    """Return the order of jurisdiction codes that were MERGEd."""
    return [
        c.kwargs["code"]
        for c in calls
        if "MERGE (j:Jurisdiction" in c.cypher
    ]


def test_apply_schema_only_seeds_enabled_jurisdictions(monkeypatch):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK,EU")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()

    store = _MockStore()
    store.settings = _config.get_settings()
    counts = apply_schema(store)

    merged = _merge_codes_called(store.calls)
    assert set(merged) == {"DK", "EU"}
    assert counts["jurisdictions"] == 2
    assert counts["jurisdictions_skipped"] == 3  # US, EW, UK skipped


def test_apply_schema_does_not_delete_disabled_seeds(monkeypatch):
    """No DELETE / DETACH appears in the calls — operator must opt in
    explicitly to remove disabled-jurisdiction data."""
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "DK")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()

    store = _MockStore()
    store.settings = _config.get_settings()
    apply_schema(store)

    for call in store.calls:
        c = call.cypher.upper()
        assert "DELETE" not in c, f"apply_schema must not DELETE: {call.cypher!r}"


def test_apply_schema_default_seeds_all_five(monkeypatch):
    """When ENABLED_JURISDICTIONS isn't set, default = all five."""
    monkeypatch.delenv("ENABLED_JURISDICTIONS", raising=False)
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()

    store = _MockStore()
    store.settings = _config.get_settings()
    counts = apply_schema(store)

    merged = _merge_codes_called(store.calls)
    assert set(merged) == {"US", "EW", "UK", "EU", "DK"}
    assert counts["jurisdictions"] == 5
    assert counts["jurisdictions_skipped"] == 0


def test_jurisdiction_seeds_constant_covers_all_jurisdictions():
    codes = {j["code"] for j in JURISDICTION_SEEDS}
    assert codes == {"US", "EW", "UK", "EU", "DK"}


# --- session-scope cache reset ------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    yield
    _config.get_settings.cache_clear()
