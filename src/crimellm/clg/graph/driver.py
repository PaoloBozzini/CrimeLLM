"""Thin Neo4j wrapper. Keeps the store behind an interface so it's swappable."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from neo4j import Driver, GraphDatabase, Session

from ..config import Settings, get_settings


class Neo4jStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._driver: Driver | None = None

    def connect(self) -> Driver:
        if self._driver is None:
            s = self.settings
            self._driver = GraphDatabase.driver(
                s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)
            )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def verify(self) -> None:
        """Raise if the DB is unreachable."""
        self.connect().verify_connectivity()

    @contextmanager
    def session(self) -> Iterator[Session]:
        drv = self.connect()
        sess = drv.session(database=self.settings.neo4j_database)
        try:
            yield sess
        finally:
            sess.close()

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.session() as s:
            return [dict(r) for r in s.run(cypher, **params)]


_store: Neo4jStore | None = None


def get_store(settings: Settings | None = None) -> Neo4jStore:
    global _store
    if _store is None:
        _store = Neo4jStore(settings)
    return _store
