"""Phase 4: Retsinformation parser + EU-directive → CELEX extraction.

Covers:
* Statute metadata + per-§/stk/nr Provisions (explode + fold modes).
* DK preamble references like "forordning (EU) 2016/679" → CELEX
  ``32016R0679`` so the IMPLEMENTS edge into the EU subgraph fires.
* Identifier helpers (Instrument id, section path, provision id).
* Unknown doc_type rejection.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from crimellm.clg.parse import retsinformation as P

FIX = Path(__file__).parent / "fixtures" / "retsinformation"
DBL_FILE = FIX / "lbk-2018-502.xml"


# --- identifier helpers ---------------------------------------------------


def test_instrument_id_shape():
    assert P.instrument_id("lbk", 2018, 502) == "dk/lbk/2018/502"


def test_section_path_variants():
    assert P.section_path("36") == "§ 36"
    assert P.section_path("36", "2") == "§ 36 stk. 2"
    assert P.section_path("36", "2", "1") == "§ 36 stk. 2 nr. 1"


def test_provision_id_variants():
    assert (
        P.provision_id("lbk", 2018, 502, "36")
        == "dk/lbk/2018/502/section/§36"
    )
    assert (
        P.provision_id("lbk", 2018, 502, "36", "2", "1")
        == "dk/lbk/2018/502/section/§36/stk.2/nr.1"
    )


# --- EU-cite → CELEX helpers ---------------------------------------------


def test_directive_to_celex_modern():
    assert P.directive_to_celex("2019", "770") == "32019L0770"


def test_directive_to_celex_two_digit_year():
    # "direktiv 95/46/EF" → 31995L0046
    assert P.directive_to_celex("95", "46") == "31995L0046"


def test_regulation_to_celex():
    assert P.regulation_to_celex("2016", "679") == "32016R0679"


def test_extract_eu_celex_refs_mixed():
    txt = (
        "Loven gennemfører Europa-Parlamentets og Rådets forordning (EU) 2016/679 "
        "samt direktiv 2016/680 og det ældre direktiv 95/46/EF. Se også raw CELEX 32019L0770."
    )
    out = P.extract_eu_celex_refs(txt)
    assert "32016R0679" in out
    assert "32016L0680" in out
    assert "31995L0046" in out
    assert "32019L0770" in out


def test_extract_eu_celex_refs_dedupes_and_orders():
    txt = (
        "forordning (EU) 2016/679, derefter direktiv 2019/770, så igen "
        "forordning (EU) 2016/679."
    )
    out = P.extract_eu_celex_refs(txt)
    assert out == ["32016R0679", "32019L0770"]


# --- statute parse: explode mode (default) -------------------------------


def test_parse_statute_metadata():
    pr = P.parse_statute_file(DBL_FILE, doc_type="lbk", year=2018, num=502)
    assert pr.instrument.id == "dk/lbk/2018/502"
    assert pr.instrument.jurisdiction == "DK"
    assert pr.instrument.year == 2018
    assert pr.instrument.short_title == "Databeskyttelsesloven"
    prov = pr.instrument.primary_provenance()
    assert prov is not None
    assert prov.source == "retsinformation"
    assert prov.source_url == "https://www.retsinformation.dk/eli/lbk/2018/502"


def test_parse_statute_provisions_explode():
    pr = P.parse_statute_file(DBL_FILE, doc_type="lbk", year=2018, num=502)
    paths = [p.section_path for p in pr.provisions]
    # § 1 has 2 stk → 2 Provisions; § 6 has 1 stk with 2 nr → 2 Provisions;
    # § 36 has 1 stk only → 1 Provision. Total: 5.
    assert len(paths) == 5
    assert "§ 1 stk. 1" in paths
    assert "§ 1 stk. 2" in paths
    assert "§ 6 stk. 1 nr. 1" in paths
    assert "§ 6 stk. 1 nr. 2" in paths
    assert "§ 36 stk. 1" in paths
    # All provisions share the publication date.
    for prov in pr.provisions:
        assert prov.valid_from == date(2018, 5, 23)
        assert prov.jurisdiction == "DK"
        assert prov.instrument_id == "dk/lbk/2018/502"


def test_parse_statute_provision_id_shape():
    pr = P.parse_statute_file(DBL_FILE, doc_type="lbk", year=2018, num=502)
    by_path = {p.section_path: p for p in pr.provisions}
    assert by_path["§ 6 stk. 1 nr. 1"].id == "dk/lbk/2018/502/section/§6/stk.1/nr.1"
    assert by_path["§ 36 stk. 1"].id == "dk/lbk/2018/502/section/§36/stk.1"


def test_parse_statute_extracts_eu_implements():
    pr = P.parse_statute_file(DBL_FILE, doc_type="lbk", year=2018, num=502)
    # Preamble cites GDPR (32016R0679) + LED (32016L0680).
    assert "32016R0679" in pr.cites_eu_celex
    assert "32016L0680" in pr.cites_eu_celex


# --- statute parse: fold mode --------------------------------------------


def test_parse_statute_fold_mode_one_per_paragraph():
    pr = P.parse_statute_file(
        DBL_FILE,
        doc_type="lbk",
        year=2018,
        num=502,
        explode_subparagraphs=False,
    )
    paths = [p.section_path for p in pr.provisions]
    # One Provision per § regardless of stk/nr granularity.
    assert paths == ["§ 1", "§ 6", "§ 36"]
    p6 = next(p for p in pr.provisions if p.section_path == "§ 6")
    # Folded text retains both nr's content + heading.
    assert "samtykke" in p6.text
    assert "Behandling af almindelige" in p6.text


# --- guards --------------------------------------------------------------


def test_parse_statute_rejects_unknown_doc_type():
    with pytest.raises(ValueError, match="doc_type"):
        P.parse_statute_file(DBL_FILE, doc_type="NOT_A_TYPE", year=2018, num=502)
