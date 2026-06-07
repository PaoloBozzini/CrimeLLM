"""Shared pytest fixtures across the whole test tree.

Anything specific to a single subtree (e.g. clg-only fixtures) stays in that
subtree's ``conftest.py``. Shared things land here.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def neo4j_store():
    """Yield a live ``Neo4jStore`` or skip the calling test.

    Lifted out of ``tests/clg/conftest.py`` so any future test tree (e.g.
    integration tests that span both pipelines) can re-use it.
    """
    from crimellm.clg.config import Settings
    from crimellm.clg.graph.driver import Neo4jStore

    store = Neo4jStore(Settings())
    try:
        store.verify()
    except Exception as e:  # noqa: BLE001 — auto-skip on any reachability error
        pytest.skip(f"Neo4j unreachable at {store.settings.neo4j_uri}: {e}")
    yield store
    store.close()
