"""Phase C.2: ``EurLexSource.fetch_one``.

Three input shapes the autofetch resolver routes here:

- CELEX directly (e.g. ``32016R0679``).
- ECLI:EU case law (e.g. ``ECLI:EU:C:2014:317``) — convert to CELEX via the
  ECLI ↔ CELEX rules (sector 6, court letter C/T/F → CJ/TJ/FJ).
- ELI slash-form (e.g. ``eu/reg/2016/679``) — convert to CELEX via the
  type-letter rules (reg → R, dir → L, dec → D).

Tests inject an ``httpx.MockTransport`` client so we exercise the real
``download_celex`` plumbing without network access.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from crimellm.clg.autofetch.exceptions import UnsupportedCite
from crimellm.clg.ingest._base import IngestContext
from crimellm.clg.ingest.eurlex import EurLexSource, celex_path


def _client_for_celex(expected_celex: str, body: bytes = b"<xml/>") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if expected_celex in str(request.url):
            return httpx.Response(200, content=body)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


@pytest.fixture
def ctx(tmp_path: Path) -> IngestContext:
    return IngestContext(raw_dir=tmp_path / "raw", interim_dir=tmp_path / "interim")


# --- happy path ------------------------------------------------------------


def test_fetch_one_celex_direct(ctx: IngestContext) -> None:
    src = EurLexSource()
    with _client_for_celex("32016R0679", body=b"<gdpr/>") as client:
        paths = src.fetch_one(ctx, cite_id="32016R0679", client=client)
    assert "32016R0679" in paths
    expected = celex_path(
        "32016R0679",
        language="en",
        fmt="fmx4",
        dest_dir=ctx.source_raw_dir("eurlex"),
    )
    assert paths["32016R0679"] == expected
    assert expected.read_bytes() == b"<gdpr/>"


def test_fetch_one_ecli_eu_converted_to_celex(ctx: IngestContext) -> None:
    # ECLI:EU:C:2014:317 → CELEX 62014CJ0317
    src = EurLexSource()
    with _client_for_celex("62014CJ0317", body=b"<digital-rights/>") as client:
        paths = src.fetch_one(ctx, cite_id="ECLI:EU:C:2014:317", client=client)
    assert paths["ECLI:EU:C:2014:317"].read_bytes() == b"<digital-rights/>"


def test_fetch_one_eli_reg_converted_to_celex(ctx: IngestContext) -> None:
    # eu/reg/2016/679 → CELEX 32016R0679 (sector 3 = legislation, R = regulation)
    src = EurLexSource()
    with _client_for_celex("32016R0679", body=b"<gdpr/>") as client:
        paths = src.fetch_one(ctx, cite_id="eu/reg/2016/679", client=client)
    assert paths["eu/reg/2016/679"].read_bytes() == b"<gdpr/>"


def test_fetch_one_eli_dir_converted_to_celex(ctx: IngestContext) -> None:
    # eu/dir/2019/770 → 32019L0770
    src = EurLexSource()
    with _client_for_celex("32019L0770", body=b"<sale-of-goods/>") as client:
        paths = src.fetch_one(ctx, cite_id="eu/dir/2019/770", client=client)
    assert paths["eu/dir/2019/770"].exists()


def test_supports_single_fetch_true() -> None:
    assert EurLexSource().supports_single_fetch() is True


# --- unsupported -----------------------------------------------------------


def test_fetch_one_unknown_shape_raises_unsupported(ctx: IngestContext) -> None:
    src = EurLexSource()
    with pytest.raises(UnsupportedCite):
        src.fetch_one(ctx, cite_id="not-a-cite")


def test_fetch_one_404_raises_unsupported(ctx: IngestContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    src = EurLexSource()
    try:
        with pytest.raises(UnsupportedCite):
            src.fetch_one(ctx, cite_id="32099R9999", client=client)
    finally:
        client.close()


# --- idempotent ------------------------------------------------------------


def test_fetch_one_second_call_uses_cache(ctx: IngestContext) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=b"<x/>")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    src = EurLexSource()
    try:
        src.fetch_one(ctx, cite_id="32016R0679", client=client)
        src.fetch_one(ctx, cite_id="32016R0679", client=client)
    finally:
        client.close()
    assert calls["n"] == 1
