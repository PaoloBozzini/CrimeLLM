"""Phase D: retrieval integration — citation misses become queue rows.

Three layers:

* ``parse_query`` extracts canonical cite ids from the question via the
  ``cite_registry`` (D.1).
* ``run_query`` checks each cite against Neo4j and enqueues the misses when
  ``autofetch_enabled`` (D.2).
* The returned ``Answer`` carries ``pending_citations`` so the CLI can flag
  "asked again in N seconds" UX (D.4).

Tests use a mock Neo4j store so the integration runs without a live
container. Phase H covers the real-Neo4j drain end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crimellm.clg.autofetch.queue import SqliteQueue
from crimellm.clg.config import Settings
from crimellm.clg.retrieval.parse_query import parse_query


# --- D.1: parse_query populates Query.citations ---------------------------


def test_parse_query_extracts_celex() -> None:
    q = parse_query("How does Regulation 32016R0679 apply here?")
    assert "32016R0679" in q.citations


def test_parse_query_extracts_ecli_eu() -> None:
    q = parse_query("Does ECLI:EU:C:2014:317 still bind?")
    assert "ECLI:EU:C:2014:317" in q.citations


def test_parse_query_extracts_eli_dk() -> None:
    q = parse_query("What does eli/lov/2020/171 require?")
    assert "eli/lov/2020/171" in q.citations


def test_parse_query_no_citations_when_plain() -> None:
    q = parse_query("Tell me about fraud sentencing in the UK")
    # No canonical cite tokens present → empty list, not None.
    assert q.citations == []


def test_parse_query_dedups_citations() -> None:
    q = parse_query(
        "Compare 32016R0679 with 32016R0679 and ECLI:EU:C:2014:317"
    )
    assert q.citations.count("32016R0679") == 1


# --- D.2: enqueue_missing_citations ----------------------------------------


@dataclass
class _MockStore:
    """Minimal Neo4j store that reports a fixed set of present cite ids."""

    present: set[str]
    calls: list[dict[str, Any]]

    def run(self, cypher: str, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append({"cypher": cypher, "kwargs": kwargs})
        ids = kwargs.get("ids") or []
        return [{"id": i} for i in ids if i in self.present]


@pytest.fixture
def queue(tmp_path: Path) -> SqliteQueue:
    q = SqliteQueue(tmp_path / "q.db")
    yield q
    q.close()


def test_enqueue_missing_skips_present_cites(queue: SqliteQueue) -> None:
    from crimellm.clg.autofetch.integration import enqueue_missing_citations

    store = _MockStore(present={"32016R0679"}, calls=[])
    pending = enqueue_missing_citations(
        ["32016R0679", "ECLI:EU:C:2014:317", "eli/lov/2020/171"],
        store=store,
        queue=queue,
    )
    # 32016R0679 already present → not pending. The other two are.
    assert sorted(pending) == sorted(["ECLI:EU:C:2014:317", "eli/lov/2020/171"])
    pending_in_queue = {row.cite_id for row in queue.list_pending()}
    assert pending_in_queue == set(pending)


def test_enqueue_missing_no_op_when_all_present(queue: SqliteQueue) -> None:
    from crimellm.clg.autofetch.integration import enqueue_missing_citations

    store = _MockStore(present={"32016R0679"}, calls=[])
    pending = enqueue_missing_citations(
        ["32016R0679"], store=store, queue=queue
    )
    assert pending == []
    assert queue.list_pending() == []


def test_enqueue_missing_skips_unresolvable_cites(queue: SqliteQueue) -> None:
    from crimellm.clg.autofetch.integration import enqueue_missing_citations

    store = _MockStore(present=set(), calls=[])
    pending = enqueue_missing_citations(
        ["totally-not-a-cite"], store=store, queue=queue
    )
    # No resolver match → cite is dropped rather than queued (otherwise the
    # worker would only ever skip them).
    assert pending == []
    assert queue.list_pending() == []


def test_enqueue_missing_handles_empty_input(queue: SqliteQueue) -> None:
    from crimellm.clg.autofetch.integration import enqueue_missing_citations

    store = _MockStore(present=set(), calls=[])
    pending = enqueue_missing_citations([], store=store, queue=queue)
    assert pending == []
    # No Neo4j call when nothing to check.
    assert store.calls == []


# --- D.4: end-to-end gating on Settings.autofetch_enabled ------------------


def test_run_query_autofetch_disabled_no_enqueue(
    tmp_path: Path, monkeypatch
) -> None:
    """Disabled autofetch: even with cite-miss in question, no queue work."""
    from crimellm.clg.autofetch import integration as I

    queue = SqliteQueue(tmp_path / "q.db")
    store = _MockStore(present=set(), calls=[])
    settings = Settings(_env_file=None)
    assert settings.autofetch_enabled is False

    pending = I.enqueue_missing_for_query(
        parse_query("Does 32016R0679 apply?"),
        store=store,
        settings=settings,
        queue_factory=lambda path: queue,
    )
    try:
        assert pending == []
        assert queue.list_pending() == []
        assert store.calls == []  # didn't even check Neo4j
    finally:
        queue.close()


def test_run_query_autofetch_enabled_enqueues(
    tmp_path: Path, monkeypatch
) -> None:
    from crimellm.clg.autofetch import integration as I

    queue = SqliteQueue(tmp_path / "q.db")
    store = _MockStore(present=set(), calls=[])
    monkeypatch.setenv("AUTOFETCH_ENABLED", "true")
    settings = Settings(_env_file=None)

    pending = I.enqueue_missing_for_query(
        parse_query("Does 32016R0679 apply?"),
        store=store,
        settings=settings,
        queue_factory=lambda path: queue,
    )
    try:
        assert "32016R0679" in pending
        assert {r.cite_id for r in queue.list_pending()} == {"32016R0679"}
    finally:
        queue.close()
