from __future__ import annotations

from datetime import date
from pathlib import Path

from crimellm.clg.parse import find_case_law as P

FIX = Path(__file__).parent / "fixtures" / "find_case_law"


def test_parse_judgment_extracts_case() -> None:
    case, refs = P.parse_judgment_file(FIX / "ewca-crim-fraud-2015.xml")
    assert "EWCA Crim 12345" in case.id
    assert case.jurisdiction == "EW"
    assert case.decision_date == date(2015, 6, 12)
    assert "Smith" in case.name
    assert case.court_id == "EWCA Crim"
    assert case.provenance and case.provenance[0].source == "find-case-law"


def test_parse_judgment_extracts_section_refs() -> None:
    _, refs = P.parse_judgment_file(FIX / "ewca-crim-fraud-2015.xml")
    keyed = {(r.instrument_id, r.section_path) for r in refs}
    assert ("uk/ukpga/2006/35", "s.2") in keyed
    assert ("uk/ukpga/2006/35", "s.3") in keyed
    assert ("uk/ukpga/1968/60", "s.1") in keyed
    assert len(refs) == 3  # dedup happens; one ref per (instrument, section)


def test_unresolvable_href_is_dropped() -> None:
    sr = P._resolve_href("https://example.com/random")
    assert sr is None


def test_resolve_href_canonical_form() -> None:
    sr = P._resolve_href("/akn/uk/act/ukpga/2006/35/section/2A")
    assert sr is not None
    assert sr.instrument_id == "uk/ukpga/2006/35"
    assert sr.section_path == "s.2A"
