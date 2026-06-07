"""Phase 1: per-jurisdiction citation parsers + registry dispatch.

Covers:
* US parser (eyecite-backed) lifts at least one ``FullCaseCitation`` from
  a realistic opinion fragment.
* DK parser hits Ufr, ECLI:DK, FED/TfK/MAD, and named-statute references.
* EU parser hits ECLI:EU, CELEX, ELI, and TFEU/TEU article refs.
* Registry round-trip: register / unregister / for_jurisdiction.
* ``enabled_jurisdictions`` gates ``parsers_for_enabled``.
"""

from __future__ import annotations

import os

import pytest

from crimellm.clg.config import Settings
from crimellm.clg.link import (
    CitationHit,
    all_parsers,
    extract_all,
    for_jurisdiction,
    parsers_for_enabled,
    register,
    registered_jurisdictions,
    unregister,
)
from crimellm.clg.link.cite_dk import DkCitationParser
from crimellm.clg.link.cite_eu import EuCitationParser
from crimellm.clg.link.cite_us import UsCitationParser


# --- registry --------------------------------------------------------------


def test_default_registrations():
    codes = registered_jurisdictions()
    assert "US" in codes
    assert "DK" in codes
    assert "EU" in codes


def test_register_unregister_roundtrip():
    class _FakeParser:
        jurisdiction = "XX"

        def extract(self, text: str) -> list[CitationHit]:
            return []

    fake = _FakeParser()
    register(fake)
    assert for_jurisdiction("XX") is fake
    assert for_jurisdiction("xx") is fake  # case-insensitive
    popped = unregister("XX")
    assert popped is fake
    assert for_jurisdiction("XX") is None


def test_parsers_for_enabled_filters():
    s = Settings(enabled_jurisdictions=["EU", "DK"])
    parsers = parsers_for_enabled(s)
    codes = {p.jurisdiction for p in parsers}
    assert "EU" in codes
    assert "DK" in codes
    assert "US" not in codes


def test_extract_all_skips_disabled_parsers():
    s = Settings(enabled_jurisdictions=["DK"])
    # US citation in the text — DK parser must not produce a US-shaped hit.
    text = "See Brown v. Board, 347 U.S. 483 (1954). Jf. U.2010.1234H."
    hits = extract_all(text, parsers=parsers_for_enabled(s))
    juris = {h.jurisdiction for h in hits}
    assert juris == {"DK"}
    assert any(h.normalised_id == "U.2010.1234.H" for h in hits)


# --- US --------------------------------------------------------------------


def test_us_parser_lifts_full_case_citation():
    parser = UsCitationParser()
    text = "See Brown v. Board of Education, 347 U.S. 483 (1954)."
    hits = parser.extract(text)
    assert hits
    assert any(h.kind == "case" and "U.S." in h.normalised_id for h in hits)
    assert all(h.jurisdiction == "US" for h in hits)


def test_us_parser_ignores_id_citation():
    parser = UsCitationParser()
    # eyecite picks up the FullCaseCitation but not the id. citation.
    text = "347 U.S. 483 (1954). Id. at 484."
    hits = parser.extract(text)
    assert all(h.kind == "case" for h in hits)
    assert len(hits) == 1


# --- DK --------------------------------------------------------------------


def test_dk_parser_ufr():
    parser = DkCitationParser()
    text = "Jf. U.2010.1234H og U.2020.456V."
    hits = parser.extract(text)
    ids = {h.normalised_id for h in hits}
    assert "U.2010.1234.H" in ids
    assert "U.2020.456.V" in ids
    assert all(h.jurisdiction == "DK" and h.kind == "case" for h in hits)


def test_dk_parser_reporters():
    parser = DkCitationParser()
    text = "Se FED 2018.1234 og TfK 2019.99 og MAD 2021.500."
    hits = parser.extract(text)
    ids = {h.normalised_id for h in hits}
    assert ids == {"FED.2018.1234", "TfK.2019.99", "MAD.2021.500"}


def test_dk_parser_ecli():
    parser = DkCitationParser()
    text = "Højesteret afgjorde sagen — ECLI:DK:HR:2023:123."
    hits = parser.extract(text)
    assert any(h.normalised_id == "ECLI:DK:HR:2023:123" for h in hits)


def test_dk_parser_statute_with_stk_nr():
    parser = DkCitationParser()
    text = "Tiltalt for overtrædelse af straffelovens § 279, stk. 2, nr. 1."
    hits = parser.extract(text)
    provisions = [h for h in hits if h.kind == "provision"]
    assert provisions
    assert provisions[0].normalised_id == "DK/straffeloven/section/279/stk.2/nr.1"


def test_dk_parser_statute_bare():
    parser = DkCitationParser()
    text = "Bestemmelsen i aftaleloven § 36 finder anvendelse."
    hits = parser.extract(text)
    assert any(h.normalised_id == "DK/aftaleloven/section/36" for h in hits)


def test_dk_parser_returns_document_order():
    parser = DkCitationParser()
    text = "U.2010.1234H — derefter straffelovens § 279 — og ECLI:DK:HR:2023:1."
    hits = parser.extract(text)
    spans = [h.span[0] for h in hits]
    assert spans == sorted(spans)


# --- EU --------------------------------------------------------------------


def test_eu_parser_ecli():
    parser = EuCitationParser()
    text = "Se Keck Mithouard, ECLI:EU:C:1993:905, og T-201/04, ECLI:EU:T:2007:289."
    hits = parser.extract(text)
    ids = {h.normalised_id for h in hits}
    assert "ECLI:EU:C:1993:905" in ids
    assert "ECLI:EU:T:2007:289" in ids
    assert all(h.kind == "case" and h.jurisdiction == "EU" for h in hits)


def test_eu_parser_celex_legislation_vs_caselaw():
    parser = EuCitationParser()
    text = "GDPR (32016R0679) og direktiv 2019/770 (32019L0770) samt 61991CJ0267."
    hits = parser.extract(text)
    by_id = {h.normalised_id: h for h in hits}
    assert by_id["32016R0679"].kind == "provision"
    assert by_id["32019L0770"].kind == "provision"
    assert by_id["61991CJ0267"].kind == "case"


def test_eu_parser_eli():
    parser = EuCitationParser()
    text = "Reference: eli/reg/2016/679 og eli/dir/2019/770."
    hits = parser.extract(text)
    ids = {h.normalised_id for h in hits}
    assert "eu/reg/2016/679" in ids
    assert "eu/dir/2019/770" in ids


def test_eu_parser_treaty_article():
    parser = EuCitationParser()
    text = "Article 101 TFEU and Art. 6 TEU and Article 263(4) TFEU."
    hits = parser.extract(text)
    ids = {h.normalised_id for h in hits}
    assert "eu/treaty/tfeu/article/101" in ids
    assert "eu/treaty/teu/article/6" in ids
    assert "eu/treaty/tfeu/article/263/para/4" in ids


# --- empty / robustness ----------------------------------------------------


def test_parsers_handle_empty_text():
    for p in all_parsers():
        assert p.extract("") == []
        assert p.extract("nothing to see here.") == [] or all(
            isinstance(h, CitationHit) for h in p.extract("nothing to see here.")
        )


def test_unknown_jurisdiction_lookup_returns_none():
    assert for_jurisdiction("ZZ") is None


# --- environment-driven gating --------------------------------------------


def test_settings_enabled_jurisdictions_from_env(monkeypatch):
    monkeypatch.setenv("ENABLED_JURISDICTIONS", "EU,DK")
    s = Settings()
    assert s.enabled_jurisdictions == ["EU", "DK"]
    parsers = parsers_for_enabled(s)
    codes = {p.jurisdiction for p in parsers}
    assert codes == {"EU", "DK"}
