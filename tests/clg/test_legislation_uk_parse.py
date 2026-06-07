from __future__ import annotations

from datetime import date
from pathlib import Path

from crimellm.clg.parse import legislation_uk as P

FIX = Path(__file__).parent / "fixtures" / "legislation_uk"


def test_parse_enacted_yields_instrument_and_provisions() -> None:
    inst, prov = P.parse_act_file(
        FIX / "ukpga-2006-35-enacted.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="enacted",
    )
    assert inst.id == "uk/ukpga/2006/35"
    assert inst.jurisdiction == "UK"
    assert "Fraud Act 2006" in inst.short_title
    assert len(prov) == 3
    ids = {p.id for p in prov}
    assert "uk/ukpga/2006/35/section/1@enacted" in ids
    assert all(p.instrument_id == "uk/ukpga/2006/35" for p in prov)
    s3 = next(p for p in prov if p.section_path == "s.3")
    assert "7 years" in s3.text
    # enacted version → valid_from inferred from the metadata date.
    assert s3.valid_from == date(2006, 11, 8)


def test_parse_current_has_different_text() -> None:
    _, prov = P.parse_act_file(
        FIX / "ukpga-2006-35-current.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="current",
    )
    s3 = next(p for p in prov if p.section_path == "s.3")
    assert "10 years" in s3.text
    # The "current" label isn't an ISO date, so valid_from stays None unless
    # provided explicitly.
    assert s3.valid_from is None


def test_explicit_valid_from_takes_precedence() -> None:
    _, prov = P.parse_act_file(
        FIX / "ukpga-2006-35-current.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="current",
        valid_from=date(2023, 1, 1),
    )
    assert all(p.valid_from == date(2023, 1, 1) for p in prov)


def test_provision_ids_are_version_scoped() -> None:
    _, enacted = P.parse_act_file(
        FIX / "ukpga-2006-35-enacted.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="enacted",
    )
    _, current = P.parse_act_file(
        FIX / "ukpga-2006-35-current.xml",
        act_type="ukpga",
        year=2006,
        number=35,
        version_label="current",
    )
    enacted_ids = {p.id for p in enacted}
    current_ids = {p.id for p in current}
    # Same section, different version_label → distinct Neo4j ids.
    assert enacted_ids.isdisjoint(current_ids)


def test_instrument_url_builder() -> None:
    from crimellm.clg.ingest.legislation_uk import act_url

    assert act_url("ukpga", 2006, 35, "current") == (
        "https://www.legislation.gov.uk/ukpga/2006/35/data.xml"
    )
    assert act_url("ukpga", 2006, 35, "enacted") == (
        "https://www.legislation.gov.uk/ukpga/2006/35/enacted/data.xml"
    )
    assert act_url("ukpga", 2006, 35, "2020-01-01") == (
        "https://www.legislation.gov.uk/ukpga/2006/35/2020-01-01/data.xml"
    )


def test_invalid_version_label_raises() -> None:
    import pytest

    from crimellm.clg.ingest.legislation_uk import LegislationUKSource

    src = LegislationUKSource(versions=("nonsense",))
    with pytest.raises(ValueError, match="invalid version label"):
        src._validate_versions()
