"""Shared pytest fixtures across the whole test tree.

Anything specific to a single subtree (e.g. clg-only fixtures) stays in that
subtree's ``conftest.py``. Shared things land here.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def neo4j_store():
    """Yield a live ``Neo4jStore`` pointed at the **test** Neo4j or skip.

    Tests in this suite DETACH DELETE nodes by label as setup. Pointing at a
    development DB wipes real data. To run them you must explicitly set
    ``CRIMELLM_TEST_NEO4J_URI`` (and optionally USER/PASSWORD/DATABASE) — see
    ``docker-compose.test.yml`` for a ready-made isolated container.

    Refuses to run if the resolved test URI matches the live ``NEO4J_URI``
    from the loaded Settings, so a misconfigured ``.env`` cannot silently
    wipe production data.
    """
    test_uri = os.environ.get("CRIMELLM_TEST_NEO4J_URI")
    if not test_uri:
        pytest.skip(
            "CRIMELLM_TEST_NEO4J_URI not set; refusing to run destructive "
            "Neo4j tests against the default URI. Start the isolated test "
            "container with `docker compose -f docker-compose.test.yml up -d` "
            "and export CRIMELLM_TEST_NEO4J_URI=bolt://localhost:7688."
        )

    from crimellm.clg.config import Settings
    from crimellm.clg.graph.driver import Neo4jStore

    live_uri = Settings().neo4j_uri
    if test_uri == live_uri:
        pytest.fail(
            f"CRIMELLM_TEST_NEO4J_URI ({test_uri}) matches the live "
            f"NEO4J_URI from Settings — tests would wipe live data. "
            f"Point the test URI at a separate container."
        )

    test_settings = Settings(
        neo4j_uri=test_uri,
        neo4j_user=os.environ.get("CRIMELLM_TEST_NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("CRIMELLM_TEST_NEO4J_PASSWORD", "crimellm-test"),
        neo4j_database=os.environ.get("CRIMELLM_TEST_NEO4J_DATABASE", "neo4j"),
    )
    store = Neo4jStore(test_settings)
    try:
        store.verify()
    except Exception as e:  # noqa: BLE001 — auto-skip on any reachability error
        pytest.skip(f"test Neo4j unreachable at {test_settings.neo4j_uri}: {e}")

    # Force any code path that calls get_store() without an explicit arg to
    # resolve to the test store, so an oversight cannot wipe live data.
    from crimellm.clg.graph import driver as _driver

    prev_store = _driver._store
    _driver._store = store
    try:
        yield store
    finally:
        _driver._store = prev_store
        store.close()
