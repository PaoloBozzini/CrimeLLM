"""Synthesizer: FakeSynthesizer round-trip + citation guard.

Also exercises ``extract_citations`` / ``check_citations``, which form the
"no fabricated citations" gate.
"""

from __future__ import annotations

from datetime import date

from crimellm.clg.retrieval.good_law import GoodLawFlag
from crimellm.clg.retrieval.parse_query import Query
from crimellm.clg.retrieval.seed import Candidate
from crimellm.clg.retrieval.synthesize import (
    DISCLAIMER,
    FakeSynthesizer,
    check_citations,
    extract_citations,
    get_synthesizer,
)


def _cand(**kw) -> Candidate:
    base = dict(
        chunk_id=None,
        text="",
        parent_type="Case",
        parent_id="cl-1",
        parent_name="Test",
        parent_jurisdiction="US",
    )
    base.update(kw)
    return Candidate(**base)  # type: ignore[arg-type]


def _query() -> Query:
    return Query(raw="What's the rule?", jurisdiction="US", as_of=date(2024, 6, 1))


def test_fake_synthesiser_cites_top_candidate() -> None:
    cands = [
        _cand(parent_id="cl-top", parent_name="Top", text="The top case's holding.", score=0.9),
        _cand(parent_id="cl-other", parent_name="Other", text="Secondary detail.", score=0.7),
    ]
    s = FakeSynthesizer()
    a = s.synthesise(query=_query(), candidates=cands, good_law={})
    assert "[cl-top]" in a.text
    assert "[cl-other]" in a.text
    assert set(a.citations) == {"cl-top", "cl-other"}
    assert a.text.startswith(DISCLAIMER)


def test_fake_synthesiser_handles_empty_context() -> None:
    a = FakeSynthesizer().synthesise(query=_query(), candidates=[], good_law={})
    assert a.citations == []
    assert "No authorities" in a.text


def test_fake_synthesiser_surfaces_caveats() -> None:
    cands = [_cand(parent_id="cl-old", parent_name="Old", text="...")]
    flags = {
        "cl-old": [
            GoodLawFlag(
                case_id="cl-old",
                treatment="overruled",
                treating_case_id="cl-new",
                treating_case_name="Newer",
            )
        ]
    }
    a = FakeSynthesizer().synthesise(query=_query(), candidates=cands, good_law=flags)
    assert a.caveats and "overruled" in a.caveats[0]
    assert "cl-new" in a.caveats[0]


def test_extract_citations_dedupes_and_preserves_order() -> None:
    text = "see [a] and [b] and [a] again, then [c]."
    assert extract_citations(text) == ["a", "b", "c"]


def test_check_citations_separates_valid_from_fabricated() -> None:
    text = "supported by [good-1] but also [made-up]."
    valid, bad = check_citations(text, {"good-1"})
    assert valid == ["good-1"]
    assert bad == ["made-up"]


def test_get_synthesizer_routes() -> None:
    assert isinstance(get_synthesizer("fake"), FakeSynthesizer)


def test_get_synthesizer_auto_falls_back_to_fake(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Force the Ollama probe to fail so the auto-pick doesn't depend on
    # whether the dev machine happens to be running an Ollama server.
    monkeypatch.setattr(
        "crimellm.clg.retrieval.synthesize._ollama_reachable",
        lambda *a, **k: False,
    )
    assert isinstance(get_synthesizer(), FakeSynthesizer)


def test_get_synthesizer_picks_ollama_when_reachable(monkeypatch) -> None:
    """If ANTHROPIC_API_KEY is missing but Ollama responds, auto-pick = ollama."""
    from crimellm.clg.retrieval.synthesize import OllamaSynthesizer

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "crimellm.clg.retrieval.synthesize._ollama_reachable",
        lambda *a, **k: True,
    )
    assert isinstance(get_synthesizer(), OllamaSynthesizer)


def test_get_synthesizer_unknown_backend_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown synthesizer"):
        get_synthesizer("nonsense")


# --- Ollama backend (mock HTTP) -------------------------------------------


def test_ollama_synthesiser_grounds_in_context(monkeypatch) -> None:
    """OllamaSynthesizer hits /api/chat and runs the citation guard."""
    from crimellm.clg.retrieval.synthesize import OllamaSynthesizer

    captured = {}

    class _FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {
                    "content": (
                        "Under UK law, fraud is punishable by up to 10 years' "
                        "imprisonment [cl-top]."
                    )
                }
            }

    def _fake_post(url, json=None, timeout=None):  # noqa: ARG001
        captured["url"] = url
        captured["payload"] = json
        return _FakeResp()

    import crimellm.clg.retrieval.synthesize as syn_mod

    class _FakeHTTPX:
        post = staticmethod(_fake_post)

        class HTTPError(Exception):
            pass

    monkeypatch.setattr(syn_mod, "httpx", _FakeHTTPX, raising=False)
    # synthesize imports httpx lazily; patch sys.modules so the lazy import sees the fake.
    import sys

    monkeypatch.setitem(sys.modules, "httpx", _FakeHTTPX)

    cands = [_cand(parent_id="cl-top", parent_name="Top", text="Fraud penalty: 10 years.")]
    s = OllamaSynthesizer(model="test-model")
    a = s.synthesise(query=_query(), candidates=cands, good_law={})

    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["model"] == "test-model"
    assert "[cl-top]" in a.text
    assert a.citations == ["cl-top"]
    # No fabricated citations -> no warning caveat.
    assert all("WARNING" not in cv for cv in a.caveats)


def test_ollama_synthesiser_flags_fabricated_citations(monkeypatch) -> None:
    """When the model emits an out-of-context id, it appears in caveats."""
    from crimellm.clg.retrieval.synthesize import OllamaSynthesizer

    class _FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": "Fabricating [made-up-id] here."}}

    class _FakeHTTPX:
        @staticmethod
        def post(url, json=None, timeout=None):  # noqa: ARG004
            return _FakeResp()

        class HTTPError(Exception):
            pass

    import sys

    monkeypatch.setitem(sys.modules, "httpx", _FakeHTTPX)

    cands = [_cand(parent_id="cl-real", parent_name="Real", text="real text")]
    s = OllamaSynthesizer(model="test-model")
    a = s.synthesise(query=_query(), candidates=cands, good_law={})

    assert a.citations == []
    assert any("made-up-id" in cv for cv in a.caveats)
