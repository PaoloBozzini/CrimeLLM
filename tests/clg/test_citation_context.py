"""Sentence-window extractor — eyecite-optional, deterministic fallback."""

from __future__ import annotations

from crimellm.clg.link.citation_context import extract_citing_sentence


def test_extract_finds_sentence_around_case_name() -> None:
    text = (
        "We agree with the appellant. Plessy v. Ferguson, 163 U.S. 537, is overruled "
        "to the extent it conflicts with this opinion. The judgment is reversed."
    )
    sent = extract_citing_sentence(text, cited_case_name="Plessy v. Ferguson")
    assert "Plessy v. Ferguson" in sent
    assert "overruled" in sent
    # Should not bleed into the unrelated final sentence about the judgment.
    assert "The judgment is reversed." not in sent


def test_extract_returns_empty_when_text_missing() -> None:
    assert extract_citing_sentence("", cited_case_name="Smith") == ""


def test_extract_returns_empty_when_no_match_and_no_eyecite_signal() -> None:
    # Nothing in this text looks like a citation OR mentions the case.
    assert (
        extract_citing_sentence(
            "The plaintiff alleged breach of contract.", cited_case_name="Smith"
        )
        == ""
    )


def test_extract_caps_max_length() -> None:
    body = "Smith. " + ("very long sentence about Smith case " * 200) + " Smith."
    sent = extract_citing_sentence(body, cited_case_name="Smith", max_chars=120)
    assert len(sent) <= 120
