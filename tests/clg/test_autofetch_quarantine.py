"""Phase F: quarantine — auto-loaded nodes tagged, eval filters them out.

Worker-loaded nodes carry ``auto_ingested=true, validated=false`` so they
stay outside the eval gold set until a human promotes them. Pre-existing
hand-loaded nodes (no flag) pass the filter via ``coalesce(n.validated,
true) = true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from typer.testing import CliRunner

from crimellm.clg.autofetch.quarantine import (
    PRESENCE_VALIDATED_CYPHER,
    mark_auto_ingested,
    promote,
)


@dataclass
class _MockStore:
    calls: list[tuple[str, dict[str, Any]]]
    rows: list[dict[str, Any]]

    def run(self, cypher: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((cypher, kwargs))
        return list(self.rows)


# --- F.1: mark_auto_ingested -----------------------------------------------


def test_mark_auto_ingested_runs_label_agnostic_match() -> None:
    store = _MockStore(calls=[], rows=[{"id": "ECLI:EU:C:2014:317"}])
    n = mark_auto_ingested("ECLI:EU:C:2014:317", store=store)
    assert n == 1
    cypher, kwargs = store.calls[-1]
    # Covers Case OR Provision OR Instrument in one MATCH.
    for label in ("Case", "Provision", "Instrument"):
        assert label in cypher
    assert "auto_ingested = true" in cypher
    assert "validated = false" in cypher
    assert kwargs["cite_id"] == "ECLI:EU:C:2014:317"


def test_mark_auto_ingested_zero_when_no_match() -> None:
    store = _MockStore(calls=[], rows=[])
    assert mark_auto_ingested("eli/lov/9999/9999", store=store) == 0


# --- F.2: validated filter Cypher -----------------------------------------


def test_presence_query_filters_unvalidated() -> None:
    """Single source of truth Cypher used by eval/retrieval boundaries."""
    assert "coalesce(n.validated, true)" in PRESENCE_VALIDATED_CYPHER
    # Without exclude_unvalidated flag, all nodes match.
    assert "$include_unvalidated" in PRESENCE_VALIDATED_CYPHER


# --- F.3: promote ----------------------------------------------------------


def test_promote_flips_validated_to_true() -> None:
    store = _MockStore(calls=[], rows=[{"id": "x", "validated": True}])
    n = promote("x", store=store)
    assert n == 1
    cypher, kwargs = store.calls[-1]
    assert "validated = true" in cypher
    assert "auto_ingested" not in cypher  # don't reset this — it's history
    assert kwargs["cite_id"] == "x"


def test_promote_zero_when_no_match() -> None:
    store = _MockStore(calls=[], rows=[])
    assert promote("nope", store=store) == 0


# --- F.3 CLI: promote command runs MERGE -----------------------------------


def test_cli_promote_calls_neo4j(monkeypatch) -> None:
    """The ``clg autofetch promote`` CLI now does real work."""
    from crimellm.clg.cli import autofetch as A

    captured: dict[str, Any] = {}

    def fake_promote(cite_id: str, *, store: Any) -> int:
        captured["cite_id"] = cite_id
        return 1

    monkeypatch.setattr("crimellm.clg.cli.autofetch.quarantine_promote", fake_promote)
    monkeypatch.setattr(
        "crimellm.clg.cli.autofetch.get_store",
        lambda: _MockStore(calls=[], rows=[]),
    )

    runner = CliRunner()
    result = runner.invoke(A.app, ["promote", "ECLI:EU:C:2014:317"])
    assert result.exit_code == 0
    assert captured["cite_id"] == "ECLI:EU:C:2014:317"
    assert "promoted 1" in result.stdout.lower()
