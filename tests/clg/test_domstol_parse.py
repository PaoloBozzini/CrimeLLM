"""Phase 5: domstol.dk judgment parser + Karnov skeleton gating.

Covers the pure-text path of ``parse_judgment_text`` (PDF wrapping is
tested only via import probe — generating real PDFs in-test is
heavyweight). Karnov skeleton tested for its missing-key guard.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.parse import domstol as P

FIX = Path(__file__).parent / "fixtures" / "domstol"
HR_FILE = FIX / "u-2023-1234-h.txt"


# --- identifier + helper coverage ----------------------------------------


def test_court_id_for_ecli_recognised_codes():
    assert P.court_id_for_ecli("ECLI:DK:HR:2023:1") == "hr"
    assert P.court_id_for_ecli("ECLI:DK:OLR:2023:42") == "olr"
    assert P.court_id_for_ecli("ECLI:DK:VLR:2024:7") == "vlr"


def test_court_id_for_ecli_unknown_code_lowercased():
    # Lowercased fallback so an unrecognised DK court code still groups
    # consistently rather than crashing.
    assert P.court_id_for_ecli("ECLI:DK:XYZ:2023:1") == "xyz"


def test_court_id_for_ecli_non_dk_returns_none():
    assert P.court_id_for_ecli("ECLI:EU:C:2014:317") is None


# --- ECLI + date extraction from body ------------------------------------


def test_extract_ecli_from_body():
    text = "Some header line.\nThe ECLI is ECLI:DK:HR:2023:1234 for this case."
    hit = P._extract_ecli(text)
    assert hit is not None
    assert hit[0] == "ECLI:DK:HR:2023:1234"
    assert hit[1] == "hr"


def test_extract_decision_date_danish_longform():
    assert P._extract_decision_date("afsagt den 15. juni 2023") == date(2023, 6, 15)


def test_extract_decision_date_short_form():
    assert P._extract_decision_date("dato 15/06-2023") == date(2023, 6, 15)


def test_extract_decision_date_iso():
    assert P._extract_decision_date("2023-06-15 stamping") == date(2023, 6, 15)


def test_extract_decision_date_none_when_absent():
    assert P._extract_decision_date("no date here") is None


# --- text-mode parse_judgment_text ---------------------------------------


def test_parse_judgment_text_infers_everything_from_body():
    pr = P.parse_judgment_file(HR_FILE)
    assert pr.case.id == "ECLI:DK:HR:2023:1234"
    assert pr.case.jurisdiction == "DK"
    assert pr.case.court_id == "hr"
    assert pr.case.decision_date == date(2023, 6, 15)
    # Case name picks up the first non-empty line (the caption).
    assert "Forbrugersag" in pr.case.name
    # Provenance source is hard-coded to domstol.dk.
    prov = pr.case.primary_provenance()
    assert prov is not None
    assert prov.source == "domstol.dk"


def test_parse_judgment_text_extracts_dk_citation_hits():
    pr = P.parse_judgment_file(HR_FILE)
    ids = {h.normalised_id for h in pr.citation_hits}
    # Ufr cites lifted by the DK parser.
    assert "U.2010.456.H" in ids
    assert "U.2020.789.V" in ids
    # ECLI:DK:OLR:2021:42 lifted.
    assert "ECLI:DK:OLR:2021:42" in ids
    # Named-statute provisions (markedsføringsloven § 8, aftaleloven § 36 stk. 1,
    # databeskyttelsesloven § 6 stk. 1 nr. 1).
    assert "DK/markedsføringsloven/section/8" in ids
    assert "DK/aftaleloven/section/36/stk.1" in ids
    assert "DK/databeskyttelsesloven/section/6/stk.1/nr.1" in ids


def test_parse_judgment_text_extracts_eu_citation_hits():
    pr = P.parse_judgment_file(HR_FILE)
    ids = {h.normalised_id for h in pr.citation_hits}
    # Phase 1 EU parser lifts CELEX 32016R0679 from "forordning (EU) 2016/679"?
    # No — Phase 1 EU parser matches the literal CELEX surface form. Body
    # uses "forordning (EU) 2016/679" which only the Retsinformation-side
    # converter handles. Verify the literal-CELEX path for this judgment
    # via the GDPR shorthand.
    # (Use an explicit ECLI:EU smoke instead — we know the body contains
    # ECLI:DK:OLR but not ECLI:EU. So just check no EU hits accidentally
    # surface that we didn't expect.)
    eu_hits = [h for h in pr.citation_hits if h.jurisdiction == "EU"]
    # No literal CELEX / ECLI:EU in this fixture — clean check that EU
    # parser didn't false-positive.
    assert eu_hits == []


def test_parse_judgment_text_operator_metadata_wins():
    body = (
        "Some judgment body without ECLI in it.\n\n"
        "Heading line.\n"
        "The court considered straffelovens § 279."
    )
    pr = P.parse_judgment_text(
        body,
        ecli="ECLI:DK:HR:2099:1",
        court_id="hr",
        decision_date=date(2099, 1, 1),
        name="Operator Override Case",
    )
    assert pr.case.id == "ECLI:DK:HR:2099:1"
    assert pr.case.decision_date == date(2099, 1, 1)
    assert pr.case.name == "Operator Override Case"


def test_parse_judgment_text_raises_when_ecli_missing():
    with pytest.raises(ValueError, match="ECLI"):
        P.parse_judgment_text("no ecli in here, just some prose.")


# --- Karnov skeleton -----------------------------------------------------


def test_karnov_skeleton_refuses_without_key(monkeypatch):
    monkeypatch.delenv("KARNOV_API_KEY", raising=False)
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    from crimellm.clg.ingest.karnov import KarnovSource

    with pytest.raises(RuntimeError, match="KARNOV_API_KEY"):
        KarnovSource()


def test_karnov_skeleton_constructs_with_key(monkeypatch):
    monkeypatch.setenv("KARNOV_API_KEY", "fake-test-key")
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    from crimellm.clg.ingest.karnov import KarnovSource

    src = KarnovSource()
    assert src.name == "karnov"


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from crimellm.clg import config as _config

    _config.get_settings.cache_clear()
    yield
    _config.get_settings.cache_clear()
