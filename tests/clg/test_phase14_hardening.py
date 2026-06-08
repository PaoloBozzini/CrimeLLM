"""Phase 14: production-hardening pickups.

Covers:
* T14.1 — ECLI:DK Case headings surface a parallel Ufr alt-id when
  ``citations`` carries one, in either the normalised ``U.YYYY.NNNN.X``
  shape or the surface ``U.YYYY.NNNNX`` shape.
* T14.2 — OCR fallback skeleton: ``parse_judgment_pdf`` tries
  ``_try_ocr_fallback`` only when the text extract is essentially empty;
  the helper returns ``None`` when ``ocrmypdf`` isn't installed (the
  ``[ocr]`` extra is optional). The non-empty path is the happy case the
  Phase 5 fixture already exercises.
* T14.3 — Language detector now distinguishes FR / DE in addition to
  DA / EN. Asymmetric signal — winner = argmax. EN remains the safe
  fallback for short / undetermined text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crimellm.clg.retrieval.prompts import (
    _pick_parallel_ufr,
    format_human_citation,
)
from crimellm.common.language import detect_language


# --- T14.1: ECLI ↔ Ufr parallel-cite -------------------------------------


def test_pick_parallel_ufr_normalised_form():
    assert _pick_parallel_ufr(["U.2010.456.H"]) == "U.2010.456H"


def test_pick_parallel_ufr_surface_form_kept():
    assert _pick_parallel_ufr(["U.2020.789V"]) == "U.2020.789V"


def test_pick_parallel_ufr_picks_first_match():
    out = _pick_parallel_ufr(
        ["not-an-ufr", "U.2023.1234.H", "U.1999.999H"]
    )
    assert out == "U.2023.1234H"


def test_pick_parallel_ufr_no_match_returns_none():
    assert _pick_parallel_ufr(["32016R0679", "random-string"]) is None
    assert _pick_parallel_ufr(None) is None
    assert _pick_parallel_ufr([]) is None


def test_format_dk_case_surfaces_ufr_parallel():
    """When the Case carries an Ufr alt-id, the heading shows both forms
    in the order DA lawyers expect: caption, Ufr, ECLI."""
    out = format_human_citation(
        "ECLI:DK:HR:2023:1234",
        parent_type="Case",
        parent_name="Forbrugersag mod Storbanken A/S",
        citations=["U.2023.1234.H"],
    )
    # Caption first, then Ufr, then ECLI, then bracketed canonical id.
    assert "Forbrugersag mod Storbanken A/S" in out
    assert "U.2023.1234H" in out
    assert "ECLI:DK:HR:2023:1234" in out
    assert "[ECLI:DK:HR:2023:1234]" in out
    # Order check: caption -> U.YYYY.NNNNH -> ECLI -> bracketed canonical.
    cap_pos = out.index("Forbrugersag")
    ufr_pos = out.index("U.2023.1234H")
    ecli_pos = out.index("ECLI:DK:HR:2023:1234 [")
    assert cap_pos < ufr_pos < ecli_pos


def test_format_dk_case_without_ufr_keeps_phase_8_behavior():
    """No ``citations`` → falls back to the Phase 8 caption + ECLI form."""
    out = format_human_citation(
        "ECLI:DK:HR:2023:1234",
        parent_type="Case",
        parent_name="Forbrugersag mod Storbanken A/S",
    )
    assert "Forbrugersag mod Storbanken A/S" in out
    assert "U." not in out  # no Ufr surfaced
    assert "[ECLI:DK:HR:2023:1234]" in out


def test_format_eu_case_ignores_dk_ufr_lookup():
    """ECLI:EU cases must NOT pick up a stray Ufr cite from citations."""
    out = format_human_citation(
        "ECLI:EU:C:1993:905",
        parent_type="Case",
        parent_name="Keck Mithouard",
        citations=["U.2023.1234.H", "61991CJ0267"],
    )
    assert "U.2023.1234" not in out
    assert "[ECLI:EU:C:1993:905]" in out


# --- T14.2: OCR fallback skeleton ----------------------------------------


def test_parse_judgment_pdf_short_extract_triggers_ocr_attempt(monkeypatch):
    """When pypdf returns near-empty text, ``parse_judgment_pdf`` calls
    the OCR fallback. The fallback returns ``None`` when ocrmypdf isn't
    installed — caller falls back to the (empty) pypdf extract."""
    from crimellm.clg.parse import domstol as P

    called: dict[str, int] = {"ocr": 0}

    def _fake_ocr(_path):
        called["ocr"] += 1
        return None  # mimic [ocr] extra not installed

    monkeypatch.setattr(P, "_try_ocr_fallback", _fake_ocr)

    class _FakePage:
        def extract_text(self):
            return ""

    class _FakeReader:
        def __init__(self, _path):
            self.pages = [_FakePage(), _FakePage()]

    monkeypatch.setattr("pypdf.PdfReader", _FakeReader)

    # ECLI is required → without OCR + without a real body, the parser
    # should still surface the explicit ECLI we pass in (parse_judgment_text
    # accepts an empty body when ecli is given).
    pr = P.parse_judgment_pdf(
        "/tmp/fake.pdf",
        ecli="ECLI:DK:HR:2099:1",
        court_id="hr",
    )
    assert called["ocr"] == 1
    assert pr.case.id == "ECLI:DK:HR:2099:1"


def test_parse_judgment_pdf_allow_ocr_false_skips_fallback(monkeypatch):
    from crimellm.clg.parse import domstol as P

    called: dict[str, int] = {"ocr": 0}
    monkeypatch.setattr(
        P,
        "_try_ocr_fallback",
        lambda _: (called.__setitem__("ocr", called["ocr"] + 1) or None),
    )

    class _FakePage:
        def extract_text(self):
            return ""

    class _FakeReader:
        def __init__(self, _path):
            self.pages = [_FakePage()]

    monkeypatch.setattr("pypdf.PdfReader", _FakeReader)

    P.parse_judgment_pdf(
        "/tmp/fake.pdf",
        ecli="ECLI:DK:HR:2099:2",
        court_id="hr",
        allow_ocr=False,
    )
    assert called["ocr"] == 0


def test_parse_judgment_pdf_non_empty_extract_skips_ocr(monkeypatch):
    """Modern PDFs with selectable text never hit the OCR path."""
    from crimellm.clg.parse import domstol as P

    called: dict[str, int] = {"ocr": 0}
    monkeypatch.setattr(
        P,
        "_try_ocr_fallback",
        lambda _: (called.__setitem__("ocr", called["ocr"] + 1) or None),
    )

    class _FakePage:
        def extract_text(self):
            return (
                "Højesterets dom afsagt den 15. juni 2023. ECLI:DK:HR:2023:1234. "
                "Sagen drejer sig om aftalelovens § 36 stk. 1 og forbrugerrettigheder."
            )

    class _FakeReader:
        def __init__(self, _path):
            self.pages = [_FakePage()]

    monkeypatch.setattr("pypdf.PdfReader", _FakeReader)

    pr = P.parse_judgment_pdf("/tmp/fake.pdf")
    assert called["ocr"] == 0
    assert pr.case.id == "ECLI:DK:HR:2023:1234"


def test_try_ocr_fallback_returns_none_when_ocrmypdf_missing(monkeypatch):
    """Forcing the optional ``ocrmypdf`` import to fail — the fallback
    must degrade gracefully, not crash the pipeline."""
    import sys

    monkeypatch.setitem(sys.modules, "ocrmypdf", None)
    from crimellm.clg.parse import domstol as P

    assert P._try_ocr_fallback("/tmp/whatever.pdf") is None


# --- T14.3: 4-way language detector --------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # DA — unchanged from Phase 7
        ("Højesteret har afsagt en afgørelse.", "da"),
        ("Loven gælder for behandling af persondata.", "da"),
        # EN — unchanged
        ("What does the Fraud Act 2006 say about phishing?", "en"),
        ("The Court has consistently held that Article 101 TFEU applies.", "en"),
        # FR — œ + ç + tion/ment suffixes + stopwords
        (
            "La Cour de justice a précisé que le règlement s'applique selon la procédure ordinaire.",
            "fr",
        ),
        (
            "Le considérant énonce les modalités d'application du règlement européen.",
            "fr",
        ),
        # DE — ß + ung/keit suffixes + stopwords
        (
            "Der Gerichtshof hat entschieden, dass die Verordnung anwendbar ist; die Auslegung erfolgt nach ständiger Rechtsprechung.",
            "de",
        ),
        (
            "Die Rechtmäßigkeit der Verarbeitung folgt aus Artikel 6 der Verordnung.",
            "de",
        ),
    ],
)
def test_detect_language_four_way(text, expected):
    lang, _ = detect_language(text)
    assert lang == expected, f"{text!r}: expected {expected}, got {lang}"


def test_detect_language_short_input_still_defaults_en():
    assert detect_language("hi")[0] == "en"
    assert detect_language("")[0] == "en"


def test_detect_language_returns_iso_639_1_among_four():
    """Contract: caller should see ``"da" | "en" | "fr" | "de"`` only."""
    samples = [
        "Højesteret",
        "The court",
        "La Cour",
        "Der Gerichtshof",
        "12345 !!!",
    ]
    for s in samples:
        lang, _ = detect_language(s)
        assert lang in {"da", "en", "fr", "de"}


def test_detect_language_eu_directive_en_stays_en():
    """EU regulations are commonly published in EN; classifier must not
    mis-route them to FR/DE just because the institutional vocabulary
    overlaps."""
    text = (
        "REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT AND OF THE "
        "COUNCIL of 27 April 2016 on the protection of natural persons "
        "with regard to the processing of personal data."
    )
    assert detect_language(text)[0] == "en"


def test_detect_language_da_judgment_with_eu_terms_stays_da():
    """A DA judgment that references EU instruments by their EN names
    must still classify as DA — the bulk of the sentence carries DA
    stopwords + diacritics."""
    text = (
        "Højesteret fastslår, at forordning (EU) 2016/679 finder anvendelse, "
        "og at den danske implementering i databeskyttelsesloven § 6 er "
        "i overensstemmelse med GDPR Article 6(1)(f)."
    )
    assert detect_language(text)[0] == "da"
