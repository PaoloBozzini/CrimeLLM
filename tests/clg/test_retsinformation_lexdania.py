"""Phase 14.5: real LexDania schema (Retsinformation API).

The live ``http://retsinformation.dk/eli/accn/<ACCN>/xml`` endpoint
returns LexDania XML — ``<Dokument>`` root, ``<Meta>`` with
``AccessionNumber`` / ``EuReferences`` / ``DiesEdicti`` / etc.,
``<DokumentIndhold>`` body with ``<Paragraf localId="N">/<Stk>/<Indentatio formaInd="Nummer">``
hierarchy. Tests cover the LexDania parser path + the inferred
``(doc_type, year, num)`` derivation used by the ingester.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.parse import retsinformation as P

FIX = Path(__file__).parent / "fixtures" / "retsinformation"
LEXDANIA = FIX / "lexdania-A20180050229.xml"


# --- end-to-end on the LexDania fixture ----------------------------------


def test_lexdania_parser_routed_on_dokument_root():
    pr = P.parse_statute_file(LEXDANIA, doc_type="lbk", year=2018, num=502)
    # Parsed via _parse_lexdania → instrument id stays the same.
    assert pr.instrument.id == "dk/lbk/2018/502"
    # Title falls back to PopularTitle when present.
    assert pr.instrument.short_title == "Databeskyttelsesloven"


def test_lexdania_metadata_from_meta_block():
    pr = P.parse_statute_file(LEXDANIA, doc_type="lbk", year=2018, num=502)
    prov = pr.instrument.primary_provenance()
    assert prov is not None
    # Source url derived from the accession number, not the slash form.
    assert prov.source_url == "http://retsinformation.dk/eli/accn/A20180050229/xml"
    assert prov.source_id == "A20180050229"
    assert prov.source == "retsinformation"


def test_lexdania_provisions_explode_mode():
    pr = P.parse_statute_file(LEXDANIA, doc_type="lbk", year=2018, num=502)
    paths = [p.section_path for p in pr.provisions]
    # § 1 has 2 stk; § 6 has 1 stk with 2 nr; § 36 has 1 stk only → 5 total.
    assert len(paths) == 5
    assert "§ 1 stk. 1" in paths
    assert "§ 1 stk. 2" in paths
    assert "§ 6 stk. 1 nr. 1" in paths
    assert "§ 6 stk. 1 nr. 2" in paths
    assert "§ 36 stk. 1" in paths


def test_lexdania_provision_id_shape():
    pr = P.parse_statute_file(LEXDANIA, doc_type="lbk", year=2018, num=502)
    by_path = {p.section_path: p for p in pr.provisions}
    assert by_path["§ 6 stk. 1 nr. 1"].id == "dk/lbk/2018/502/section/§6/stk.1/nr.1"
    assert by_path["§ 36 stk. 1"].id == "dk/lbk/2018/502/section/§36/stk.1"


def test_lexdania_valid_from_from_dies_edicti():
    pr = P.parse_statute_file(LEXDANIA, doc_type="lbk", year=2018, num=502)
    for p in pr.provisions:
        # DiesEdicti=2018-05-24 wins over StartDate fallback.
        assert p.valid_from == date(2018, 5, 24)


def test_lexdania_eu_references_extracted_directly():
    """LexDania <Meta><EuReferences> gives CELEX strings verbatim — no
    DK-preamble regex needed. Phase 4 IMPLEMENTS seeds come from this."""
    pr = P.parse_statute_file(LEXDANIA, doc_type="lbk", year=2018, num=502)
    assert "32016R0679" in pr.cites_eu_celex
    assert "32016L0680" in pr.cites_eu_celex


def test_lexdania_fold_mode_one_per_paragraph():
    pr = P.parse_statute_file(
        LEXDANIA, doc_type="lbk", year=2018, num=502, explode_subparagraphs=False
    )
    paths = [p.section_path for p in pr.provisions]
    assert paths == ["§ 1", "§ 6", "§ 36"]


# --- _infer_slash_form helper (ingester resolver) ------------------------


def test_infer_slash_form_from_real_lexdania():
    from crimellm.clg.ingest.retsinformation import _infer_slash_form

    triple = _infer_slash_form(LEXDANIA)
    assert triple == ("lbk", 2018, 502)


def test_infer_slash_form_returns_none_on_synthetic_schema():
    """Synthetic fixture uses lowercase <dokument> — the LexDania helper
    must decline so the caller can fall back."""
    from crimellm.clg.ingest.retsinformation import _infer_slash_form

    triple = _infer_slash_form(FIX / "lbk-2018-502.xml")
    assert triple is None


# --- URL builders --------------------------------------------------------


def test_accn_xml_url():
    from crimellm.clg.ingest.retsinformation import accn_xml_url

    assert (
        accn_xml_url("A20180050229")
        == "http://retsinformation.dk/eli/accn/A20180050229/xml"
    )


def test_discover_url_with_and_without_date():
    from crimellm.clg.ingest.retsinformation import discover_url

    assert discover_url() == "https://api.retsinformation.dk/v1/Documents"
    assert (
        discover_url("2026-06-05")
        == "https://api.retsinformation.dk/v1/Documents?date=2026-06-05"
    )


def test_accn_cache_path(tmp_path):
    from crimellm.clg.ingest.retsinformation import accn_cache_path

    p = accn_cache_path("A20180050229", tmp_path)
    assert p.name == "A20180050229.xml"
    assert p.parent == tmp_path
