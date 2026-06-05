"""Shared fixtures: skip Neo4j-dependent tests when the DB is unreachable."""
from __future__ import annotations

import pytest

from crimellm.clg.config import Settings
from crimellm.clg.graph.driver import Neo4jStore


@pytest.fixture(scope="session")
def neo4j_store() -> Neo4jStore:
    store = Neo4jStore(Settings())
    try:
        store.verify()
    except Exception as e:
        pytest.skip(f"Neo4j unreachable at {store.settings.neo4j_uri}: {e}")
    yield store
    store.close()
