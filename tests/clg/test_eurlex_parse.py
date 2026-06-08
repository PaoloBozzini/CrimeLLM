"""Phase 3: EUR-Lex Akoma Ntoso parser.

Covers:
* Regulation (legislation): Instrument metadata + per-article Provisions +
  IMPLEMENTS-seed CELEX extraction from preamble + authorialNote.
* Judgment: Case metadata + ECLI/CELEX co-citation extraction.
* Identifier helpers (CELEX → Instrument id, ECLI → Case id, sector kind).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.parse import eurlex as P

FIX = Path(__file__).parent / "fixtures" / "eurlex"
GDPR_FILE = FIX / "32016R0679.en.fmx4.xml"
DIRECTIVE_FILE = FIX / "32019L0770.en.fmx4.xml"
JUDGMENT_FILE = FIX / "62012CJ0131.en.fmx4.xml"


# --- identifier helpers ---------------------------------------------------


def test_instrument_id_shape():
    assert P.instrument_id_from_celex("32016R0679") == "eu/celex/32016R0679"


def test_provision_id_shape():
    assert P.provision_id("32016R0679", "art.6") == "eu/celex/32016R0679/article/art.6"


def test_case_id_passthrough():
    assert P.case_id_from_ecli("ECLI:EU:C:2014:317") == "ECLI:EU:C:2014:317"


def test_celex_kind_sectors():
    assert P.celex_kind("32016R0679") == "legislation"
    assert P.celex_kind("62012CJ0131") == "case"
    assert P.celex_kind("11997M") == "other"
    assert P.celex_kind("") == "other"


# --- regulation parse -----------------------------------------------------


def test_parse_regulation_metadata():
    pr = P.parse_regulation_file(GDPR_FILE)
    assert pr.instrument.id == "eu/celex/32016R0679"
    assert pr.instrument.jurisdiction == "EU"
    assert pr.instrument.year == 2016
    assert "REGULATION" in pr.instrument.short_title
    # Provenance populated.
    prov = pr.instrument.primary_provenance()
    assert prov is not None
    assert prov.source == "eur-lex"
    assert prov.source_id == "32016R0679@en"


def test_parse_regulation_provisions():
    pr = P.parse_regulation_file(GDPR_FILE)
    paths = [p.section_path for p in pr.provisions]
    assert paths == ["art.1", "art.6"]
    # Article 1 text includes both paragraphs + heading.
    art1 = next(p for p in pr.provisions if p.section_path == "art.1")
    assert "Subject-matter" in art1.text
    assert "personal data" in art1.text
    assert art1.valid_from == date(2016, 4, 27)
    assert art1.version_id == "en"
    assert art1.instrument_id == "eu/celex/32016R0679"


def test_parse_regulation_extracts_implements_seeds():
    pr = P.parse_regulation_file(GDPR_FILE)
    # Preamble + footnote mention three other CELEX ids; self-cite excluded.
    assert "31995L0046" in pr.cites_celex
    assert "32001R0045" in pr.cites_celex
    assert "32002L0058" in pr.cites_celex
    assert "32016R0679" not in pr.cites_celex


# --- Phase 12: directive fixture rounds out EU coverage ------------------


def test_parse_directive_metadata():
    pr = P.parse_regulation_file(DIRECTIVE_FILE)
    assert pr.instrument.id == "eu/celex/32019L0770"
    assert pr.instrument.jurisdiction == "EU"
    assert pr.instrument.year == 2019
    assert "DIRECTIVE (EU) 2019/770" in pr.instrument.short_title


def test_parse_directive_provisions():
    pr = P.parse_regulation_file(DIRECTIVE_FILE)
    paths = [p.section_path for p in pr.provisions]
    assert paths == ["art.3", "art.8"]
    art8 = next(p for p in pr.provisions if p.section_path == "art.8")
    assert "Conformity" in art8.text


def test_parse_directive_implements_seeds():
    pr = P.parse_regulation_file(DIRECTIVE_FILE)
    # Directive 2011/83 + GDPR + sibling 2019/771 cited, self-cite excluded.
    assert "32011L0083" in pr.cites_celex
    assert "32016R0679" in pr.cites_celex
    assert "32019L0771" in pr.cites_celex
    assert "32019L0770" not in pr.cites_celex


def test_parse_regulation_explicit_celex_override():
    """When operator passes --celex it wins over the FRBRalias value."""
    pr = P.parse_regulation_file(GDPR_FILE, celex="32016R0679")
    assert pr.instrument.id == "eu/celex/32016R0679"


def test_parse_regulation_missing_celex_raises(tmp_path):
    no_celex = tmp_path / "no_celex.xml"
    no_celex.write_text(
        '<?xml version="1.0"?>'
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        "<act><meta><identification><FRBRWork>"
        '<FRBRdate date="2020-01-01" name="adoption"/>'
        "</FRBRWork></identification></meta></act>"
        "</akomaNtoso>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CELEX"):
        P.parse_regulation_file(no_celex)


# --- judgment parse -------------------------------------------------------


def test_parse_judgment_metadata():
    jp = P.parse_judgment_file(JUDGMENT_FILE)
    assert jp.case.id == "ECLI:EU:C:2014:317"
    assert jp.case.jurisdiction == "EU"
    assert jp.case.decision_date == date(2014, 5, 13)
    assert "Google Spain" in jp.case.name
    # CELEX surfaces in citations[] when present.
    assert "62012CJ0131" in jp.case.citations
    assert jp.case.court_id == "cjeu"


def test_parse_judgment_extracts_co_citations():
    jp = P.parse_judgment_file(JUDGMENT_FILE)
    assert "ECLI:EU:C:2003:294" in jp.cites_ecli
    assert "ECLI:EU:C:1995:411" in jp.cites_ecli
    assert "ECLI:EU:C:2014:317" not in jp.cites_ecli  # self-cite excluded
    assert "32001L0095" in jp.cites_celex
    assert "32016R0679" in jp.cites_celex
    assert "62012CJ0131" not in jp.cites_celex  # self-cite excluded


def test_parse_judgment_missing_ecli_raises(tmp_path):
    no_ecli = tmp_path / "no_ecli.xml"
    no_ecli.write_text(
        '<?xml version="1.0"?>'
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        "<judgment><meta><identification><FRBRWork>"
        '<FRBRdate date="2020-01-01" name="judgment"/>'
        "</FRBRWork></identification></meta></judgment>"
        "</akomaNtoso>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ECLI"):
        P.parse_judgment_file(no_ecli)


# --- regression: same parser handles unversioned AKN namespace ------------


def test_parse_handles_unversioned_akn_namespace(tmp_path):
    f = tmp_path / "unversioned.xml"
    f.write_text(
        '<?xml version="1.0"?>'
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn">'
        "<act><meta><identification><FRBRWork>"
        '<FRBRalias name="CELEX" value="32020R0001"/>'
        '<FRBRdate date="2020-01-01" name="adoption"/>'
        "</FRBRWork></identification></meta>"
        "<preface><docTitle>Tiny Regulation</docTitle></preface>"
        "<body><article><num>Article 1</num><content><p>Hello.</p></content></article></body>"
        "</act></akomaNtoso>",
        encoding="utf-8",
    )
    pr = P.parse_regulation_file(f)
    assert pr.instrument.id == "eu/celex/32020R0001"
    assert len(pr.provisions) == 1
    assert pr.provisions[0].section_path == "art.1"
