"""Chunker — natural unit splitting."""

from __future__ import annotations

from datetime import date

from crimellm.clg.embed.chunker import (
    DEFAULT_MAX_CHARS,
    chunk_case,
    chunk_provision,
    iter_chunks,
)
from crimellm.clg.models import Case, Provision


def _provision(text: str, sec: str = "s.1") -> Provision:
    return Provision(
        id=f"uk/ukpga/2006/35/section/{sec}@test",
        instrument_id="uk/ukpga/2006/35",
        jurisdiction="UK",
        section_path=sec,
        text=text,
        version_id="test",
    )


def test_short_provision_emits_single_chunk() -> None:
    prov = _provision("A short section about fraud.")
    chunks = chunk_provision(prov)
    assert len(chunks) == 1
    assert chunks[0].parent_type == "Provision"
    assert chunks[0].parent_id == prov.id
    assert chunks[0].text == "A short section about fraud."


def test_provision_chunk_ids_are_content_hashes() -> None:
    a = chunk_provision(_provision("same text"))[0]
    b = chunk_provision(_provision("same text", sec="s.2"))[0]
    # Different parent, same text -> same chunk id (content-addressed).
    assert a.id == b.id


def test_long_provision_gets_windowed() -> None:
    body = "fraud sentence. " * 500  # ~8 000 chars
    chunks = chunk_provision(_provision(body), max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)
    # Overlap: adjacent windows share at least the start of the second.
    assert chunks[1].text[:50] in chunks[0].text or len(chunks[1].text) <= 200


def test_empty_provision_yields_nothing() -> None:
    assert chunk_provision(_provision("   ")) == []


def test_chunk_case_splits_paragraphs() -> None:
    case = Case(
        id="cl-cluster-2001",
        jurisdiction="US",
        court_id="scotus",
        name="Chevron v NRDC",
        decision_date=date(1984, 6, 25),
    )
    body = (
        "First paragraph.\n\nSecond paragraph about deference.\n\nThird, on agency interpretation."
    )
    chunks = chunk_case(case, body)
    assert len(chunks) == 3
    assert chunks[0].text.startswith("First")
    assert chunks[1].parent_id == case.id


def test_iter_chunks_dispatches_by_type() -> None:
    prov = _provision("section text")
    case = Case(
        id="cl-cluster-9",
        jurisdiction="US",
        court_id="scotus",
        name="Test",
        decision_date=None,
    )
    out = list(iter_chunks([prov, (case, "para one.\n\npara two.")]))
    by_parent = {c.parent_type for c in out}
    assert by_parent == {"Provision", "Case"}
    assert len(out) == 3  # 1 provision + 2 case paragraphs


def test_default_max_chars_is_reasonable() -> None:
    """A typical UK section is well under DEFAULT_MAX_CHARS so it stays one chunk."""
    assert DEFAULT_MAX_CHARS >= 1000
