"""Phase C.1: ``RetsinformationSource.fetch_one``.

The autofetch worker hands cite ids like ``eli/lov/2020/171`` to the source.
Retsinformation needs to:

1. Recognise the slash-form ELI and pull (doc_type, year, num).
2. Resolve to an accession number via the RDFa lookup endpoint.
3. Download the LexDania XML for that accn into the raw cache.
4. Return ``{cite_id: cached_path}``.

Slug shape ids (``DK/<short_title>/section/279``) hit ``UnsupportedCite``
because there is no name → accn table. An operator who needs them must add
the slug → ELI mapping manually before re-enqueueing.

Tests use ``httpx.MockTransport`` via the injectable ``client`` kwarg, so
we exercise the real ``download_accn`` + ``resolve_accn`` plumbing without
touching the network and without monkeypatching the global ``httpx`` module.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from crimellm.clg.autofetch.exceptions import UnsupportedCite
from crimellm.clg.ingest._base import IngestContext
from crimellm.clg.ingest.retsinformation import (
    RetsinformationSource,
    accn_cache_path,
)


def _rdfa_response_for(accn: str) -> bytes:
    """Synthesise an RDFa payload >= 4 KB containing the accn literal."""
    pad = b"x" * 5000
    return (
        b'<html><body>' + pad
        + b' "' + accn.encode() + b'" '
        + b'</body></html>'
    )


def _xml_response_for(accn: str) -> bytes:
    return (
        '<?xml version="1.0"?><Dokument><Meta>'
        '<DocumentType>LBK H</DocumentType>'
        '<Year>2020</Year><Number>171</Number>'
        '</Meta></Dokument>'
    ).encode()


def _client_for_eli(year: int, num: int, accn: str) -> httpx.Client:
    """Build a real httpx.Client wired to a MockTransport for one (year, num) → accn."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if f"/eli/lta/{year}/{num}.rdfa" in url:
            return httpx.Response(200, content=_rdfa_response_for(accn))
        if f"/eli/accn/{accn}/xml" in url:
            return httpx.Response(200, content=_xml_response_for(accn))
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


@pytest.fixture
def ctx(tmp_path: Path) -> IngestContext:
    return IngestContext(raw_dir=tmp_path / "raw", interim_dir=tmp_path / "interim")


# --- happy path ------------------------------------------------------------


def test_fetch_one_eli_lov_downloads_xml(ctx: IngestContext) -> None:
    src = RetsinformationSource()
    with _client_for_eli(2020, 171, "A20200017105") as client:
        paths = src.fetch_one(ctx, cite_id="eli/lov/2020/171", client=client)

    assert "eli/lov/2020/171" in paths
    expected_path = accn_cache_path(
        "A20200017105", ctx.source_raw_dir("retsinformation")
    )
    assert paths["eli/lov/2020/171"] == expected_path
    assert expected_path.exists()
    assert b"Dokument" in expected_path.read_bytes()


def test_fetch_one_eli_lbk_supported(ctx: IngestContext) -> None:
    src = RetsinformationSource()
    with _client_for_eli(2018, 1156, "A20181115605") as client:
        paths = src.fetch_one(ctx, cite_id="eli/lbk/2018/1156", client=client)
    assert paths["eli/lbk/2018/1156"].exists()


def test_supports_single_fetch_true() -> None:
    assert RetsinformationSource().supports_single_fetch() is True


# --- unsupported / errors --------------------------------------------------


def test_fetch_one_slug_shape_raises_unsupported(ctx: IngestContext) -> None:
    src = RetsinformationSource()
    with pytest.raises(UnsupportedCite) as ei:
        src.fetch_one(ctx, cite_id="DK/straffeloven/section/279")
    assert "slug" in str(ei.value).lower() or "short title" in str(ei.value).lower()


def test_fetch_one_unknown_shape_raises_unsupported(ctx: IngestContext) -> None:
    src = RetsinformationSource()
    with pytest.raises(UnsupportedCite):
        src.fetch_one(ctx, cite_id="totally-not-an-eli")


def test_fetch_one_eli_not_resolvable_raises_unsupported(ctx: IngestContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if ".rdfa" in str(request.url):
            # < 4 KB → resolve_accn returns None (the SPA empty-shell case).
            return httpx.Response(200, content=b"<html>tiny shell</html>")
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    src = RetsinformationSource()
    try:
        with pytest.raises(UnsupportedCite):
            src.fetch_one(ctx, cite_id="eli/lov/9999/9999", client=client)
    finally:
        client.close()


# --- idempotent ------------------------------------------------------------


def test_fetch_one_second_call_uses_cache(ctx: IngestContext) -> None:
    call_count = {"resolve": 0, "xml": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if ".rdfa" in url:
            call_count["resolve"] += 1
            return httpx.Response(200, content=_rdfa_response_for("A20200017105"))
        if "/xml" in url:
            call_count["xml"] += 1
            return httpx.Response(200, content=_xml_response_for("A20200017105"))
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    src = RetsinformationSource()
    try:
        src.fetch_one(ctx, cite_id="eli/lov/2020/171", client=client)
        src.fetch_one(ctx, cite_id="eli/lov/2020/171", client=client)
    finally:
        client.close()
    # Second call hits the on-disk cache for the XML body. Resolver still
    # runs (cheap RDFa GET) but the heavy XML download stays at 1.
    assert call_count["xml"] == 1
