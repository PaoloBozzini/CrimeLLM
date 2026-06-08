"""Phase C.4: ``LegislationUKSource.fetch_one``.

UK ELI shape: ``uk/<act_type>/<year>/<num>[/...]``. Worker fetches the
"current" version by default — point-in-time versions are an operator
concern (Phase 6 enrichment), not the worker's auto-coverage path.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from crimellm.clg.autofetch.exceptions import UnsupportedCite
from crimellm.clg.ingest._base import IngestContext
from crimellm.clg.ingest.legislation_uk import LegislationUKSource, act_path


def _client(expected_path_fragment: str, body: bytes = b"<Act/>") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if expected_path_fragment in str(request.url):
            return httpx.Response(200, content=body)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


@pytest.fixture
def ctx(tmp_path: Path) -> IngestContext:
    return IngestContext(raw_dir=tmp_path / "raw", interim_dir=tmp_path / "interim")


def test_fetch_one_ukpga_current_version(ctx: IngestContext) -> None:
    src = LegislationUKSource()
    with _client("/ukpga/2018/12/data.xml", body=b"<DataProtectionAct/>") as client:
        paths = src.fetch_one(ctx, cite_id="uk/ukpga/2018/12", client=client)
    assert "uk/ukpga/2018/12" in paths
    expected = act_path(
        "ukpga", 2018, 12, "current", ctx.source_raw_dir("legislation_uk")
    )
    assert paths["uk/ukpga/2018/12"] == expected
    assert expected.read_bytes() == b"<DataProtectionAct/>"


def test_fetch_one_handles_extra_path_segments(ctx: IngestContext) -> None:
    src = LegislationUKSource()
    with _client("/ukpga/2006/35/data.xml") as client:
        paths = src.fetch_one(
            ctx, cite_id="uk/ukpga/2006/35/section/1", client=client
        )
    assert paths["uk/ukpga/2006/35/section/1"].exists()


def test_supports_single_fetch_true() -> None:
    assert LegislationUKSource().supports_single_fetch() is True


def test_fetch_one_unknown_shape_raises_unsupported(ctx: IngestContext) -> None:
    src = LegislationUKSource()
    with pytest.raises(UnsupportedCite):
        src.fetch_one(ctx, cite_id="not-an-eli")


def test_fetch_one_404_raises_unsupported(ctx: IngestContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    src = LegislationUKSource()
    try:
        with pytest.raises(UnsupportedCite):
            src.fetch_one(ctx, cite_id="uk/ukpga/9999/9999", client=client)
    finally:
        client.close()


def test_fetch_one_second_call_uses_cache(ctx: IngestContext) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=b"<Act/>")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    src = LegislationUKSource()
    try:
        src.fetch_one(ctx, cite_id="uk/ukpga/2018/12", client=client)
        src.fetch_one(ctx, cite_id="uk/ukpga/2018/12", client=client)
    finally:
        client.close()
    assert calls["n"] == 1
