"""Phase 14.5: slash-form ELI → accession-number resolver.

The Retsinformation API uses accession numbers as the canonical id, but
DK lawyers cite the slash-form ELI (``lov/2018/502``). ``resolve_accn``
bridges the gap via the SPA's RDFa endpoint:

    https://www.retsinformation.dk/eli/lta/<year>/<num>.rdfa

The ``lta`` (Lovtidende A) publication media covers LOV, LBK, and BEK —
the conventional ``lov`` / ``lbk`` / ``bek`` slash-forms return the
empty SPA shell so the resolver routes via ``lta`` regardless of the
operator's slug choice.

Tests mock the HTTP layer so the suite stays network-free.
"""

from __future__ import annotations

import pytest

from crimellm.clg.ingest import retsinformation as R


_RDFA_BODY_WITH_ACCN = (
    b'<html xmlns:eli="http://data.europa.eu/eli/ontology#">\n'
    b'<head><meta property="eli:id_local" content="A20180050230"/></head>\n'
    b'<body><div about="A20180050230">'
    + b"x" * 5_000  # bump past the 4 KB SPA-shell threshold
    + b'\n  some prose mentioning "A20180050230" once more\n'
    b'</div></body></html>'
)

_RDFA_EMPTY_SPA_SHELL = b'<!doctype html><html><div id="root"></div></html>'  # ~50 bytes


def test_is_accn_recognises_canonical_form():
    assert R.is_accn("A20180050230")
    assert R.is_accn("B20260050805")
    assert R.is_accn("C20260935909")
    assert R.is_accn("D20240012345")


def test_is_accn_rejects_non_accn():
    assert not R.is_accn("lov/2018/502")
    assert not R.is_accn("32016R0679")
    assert not R.is_accn("A2018005023")  # 10 chars after prefix, not 11
    assert not R.is_accn("E20180050230")  # unknown prefix


def test_is_slash_form_recognises_eli():
    assert R.is_slash_form("lov/2018/502")
    assert R.is_slash_form("lbk/2024/434")
    assert R.is_slash_form("bek/2026/508")
    assert R.is_slash_form("lta/2018/502")


def test_is_slash_form_rejects_other():
    assert not R.is_slash_form("A20180050230")
    assert not R.is_slash_form("lov/abc/def")
    assert not R.is_slash_form("/lov/2018/502")


def test_rdfa_url_uses_lta_by_default():
    assert R.rdfa_url(2018, 502) == "https://www.retsinformation.dk/eli/lta/2018/502.rdfa"


def test_rdfa_url_custom_pub_media():
    assert R.rdfa_url(2024, 42, pub_media="ltc") == (
        "https://www.retsinformation.dk/eli/ltc/2024/42.rdfa"
    )


# --- resolve_accn -------------------------------------------------------


def test_resolve_accn_extracts_from_rdfa(monkeypatch):
    class _R:
        content = _RDFA_BODY_WITH_ACCN
        status_code = 200

    def _fake_get(_client, _url, **_kw):
        return _R()

    monkeypatch.setattr(R, "get_with_retry", _fake_get)
    assert R.resolve_accn(2018, 502) == "A20180050230"


def test_resolve_accn_returns_none_on_spa_shell(monkeypatch):
    class _R:
        content = _RDFA_EMPTY_SPA_SHELL
        status_code = 200

    def _fake_get(_client, _url, **_kw):
        return _R()

    monkeypatch.setattr(R, "get_with_retry", _fake_get)
    assert R.resolve_accn(9999, 99999) is None


def test_resolve_accn_returns_none_on_404(monkeypatch):
    import httpx

    def _fake_get(_client, _url, **_kw):
        request = httpx.Request("GET", "https://example/r.rdfa")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("404", request=request, response=response)

    monkeypatch.setattr(R, "get_with_retry", _fake_get)
    assert R.resolve_accn(2018, 502) is None


# --- normalise_items ----------------------------------------------------


def test_normalise_items_passes_through_accn(monkeypatch):
    monkeypatch.setattr(
        R, "resolve_accn", lambda *a, **kw: pytest.fail("must not be called")
    )
    out = R.normalise_items(["A20180050230"])
    assert out == [("A20180050230", None)]


def test_normalise_items_resolves_slash_form(monkeypatch):
    calls: list[tuple[int, int]] = []

    def _fake_resolve(year, num, **_kw):
        calls.append((year, num))
        return "A20180050230"

    monkeypatch.setattr(R, "resolve_accn", _fake_resolve)
    out = R.normalise_items(["lov/2018/502"])
    assert out == [("A20180050230", ("lov", 2018, 502))]
    assert calls == [(2018, 502)]


def test_normalise_items_mixes_forms(monkeypatch):
    def _fake_resolve(year, num, **_kw):
        return f"A{year:04d}00{num:05d}"

    monkeypatch.setattr(R, "resolve_accn", _fake_resolve)
    out = R.normalise_items(["A20180050230", "lbk/2024/434", "B20260050805"])
    assert out == [
        ("A20180050230", None),
        ("A20240000434", ("lbk", 2024, 434)),
        ("B20260050805", None),
    ]


def test_normalise_items_raises_on_resolve_miss(monkeypatch):
    monkeypatch.setattr(R, "resolve_accn", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="could not resolve"):
        R.normalise_items(["lov/9999/99999"])


def test_normalise_items_raises_on_unrecognised_entry(monkeypatch):
    monkeypatch.setattr(R, "resolve_accn", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="unrecognised --items entry"):
        R.normalise_items(["32016R0679"])  # CELEX, not an accn or slash-form


def test_normalise_items_skips_empty(monkeypatch):
    monkeypatch.setattr(
        R, "resolve_accn", lambda *a, **kw: pytest.fail("must not be called")
    )
    out = R.normalise_items(["", "  ", "A20180050230"])
    assert out == [("A20180050230", None)]
