"""Phase A.4: Source ABC widening for single-ID fetch.

The autofetch worker needs to grab one doc by canonical ID instead of a full
bulk dump. Phase A only extends the ABC with the contract — per-source
overrides land in Phase C. These tests pin the default behaviour so a source
that forgets to override fails loud at the worker boundary, not silently.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from crimellm.clg.ingest._base import IngestContext, LoadReport, Source


class _StubSource(Source):
    """Minimal Source — does not override single-fetch methods."""

    name = "stub"

    def download(self, ctx: IngestContext) -> dict[str, Path]:  # pragma: no cover - unused
        return {}

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:  # pragma: no cover
        yield from ()

    def load(self, ctx: IngestContext) -> LoadReport:  # pragma: no cover
        return LoadReport(source=self.name)


def test_source_default_does_not_support_single_fetch() -> None:
    src = _StubSource()
    assert src.supports_single_fetch() is False


def test_source_default_fetch_one_raises() -> None:
    src = _StubSource()
    ctx = IngestContext()
    with pytest.raises(NotImplementedError):
        src.fetch_one(ctx, cite_id="ECLI:DK:HR:2020:1")


def test_source_subclass_can_override_single_fetch(tmp_path: Path) -> None:
    """An override flips ``supports_single_fetch`` and returns a path map."""

    class _SingleFetchSource(_StubSource):
        name = "single"

        def supports_single_fetch(self) -> bool:
            return True

        def fetch_one(self, ctx: IngestContext, cite_id: str) -> dict[str, Path]:
            out = tmp_path / f"{cite_id}.json"
            out.write_text("{}")
            return {cite_id: out}

    src = _SingleFetchSource()
    assert src.supports_single_fetch() is True
    paths = src.fetch_one(IngestContext(), cite_id="ECLI:DK:HR:2020:1")
    assert "ECLI:DK:HR:2020:1" in paths
    assert paths["ECLI:DK:HR:2020:1"].exists()
