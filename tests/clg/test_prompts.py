"""Phase 8: language-routed prompts + citation format helpers.

Covers:
* Disclaimer + system prompt selection by language code (case-insensitive,
  EN fallback for unknown).
* DA system prompt avoids common-law-specific terms.
* Citation format enrichment for DK lbk Provisions, EU CELEX articles,
  ECLI passthrough, UK statute back-compat.
* FakeSynthesizer respects ``query.language`` for both the disclaimer
  and the body prose.
"""

from __future__ import annotations

from datetime import date

import pytest

from crimellm.clg.retrieval.prompts import (
    DISCLAIMER_DA,
    DISCLAIMER_EN,
    SYSTEM_PROMPT_DA,
    SYSTEM_PROMPT_EN,
    disclaimer_for,
    format_candidates_block,
    format_human_citation,
    no_context_message_for,
    system_prompt_for,
)


# --- disclaimer / system prompt selection --------------------------------


def test_disclaimer_for_en():
    assert disclaimer_for("en") == DISCLAIMER_EN


def test_disclaimer_for_da():
    out = disclaimer_for("da")
    assert out == DISCLAIMER_DA
    # Sanity: text actually is Danish.
    assert "juridisk rådgivning" in out
    assert "Verificer alle citater" in out


def test_disclaimer_case_insensitive():
    assert disclaimer_for("DA") == DISCLAIMER_DA
    assert disclaimer_for("EN") == DISCLAIMER_EN


def test_disclaimer_unknown_falls_back_to_en():
    assert disclaimer_for("xx") == DISCLAIMER_EN
    assert disclaimer_for(None) == DISCLAIMER_EN
    assert disclaimer_for("") == DISCLAIMER_EN


def test_system_prompt_for_en_is_common_law():
    p = system_prompt_for("en")
    assert p is SYSTEM_PROMPT_EN
    # Common-law vocabulary present.
    assert "primary-law" in p


def test_system_prompt_for_da_is_civil_law():
    p = system_prompt_for("da")
    assert p is SYSTEM_PROMPT_DA
    # DA prompt should reference DK civil-law concepts.
    assert "praksis" in p
    assert "Højesteret" in p
    # DA prompt mentions "binding precedent" only contrastively
    # ("ikke om binding precedent") — it must explicitly warn against
    # the common-law frame.
    assert "ikke om" in p and "binding precedent" in p
    # DA prompt does NOT use English "still good law" phrasing as
    # standalone instruction (which would push a common-law analysis).
    assert "still good law" not in p
    # DA disclaimer/instructions are in Danish.
    assert "Strenge regler" in p
    assert "Brug UDELUKKENDE" in p


def test_system_prompt_unknown_falls_back_to_en():
    assert system_prompt_for("zz") is SYSTEM_PROMPT_EN
    assert system_prompt_for(None) is SYSTEM_PROMPT_EN


def test_no_context_message_localised():
    assert "Ingen kilder" in no_context_message_for("da")
    assert "No authorities" in no_context_message_for("en")
    assert "No authorities" in no_context_message_for(None)


# --- format_human_citation ------------------------------------------------


def test_format_dk_provision_with_stk_and_nr():
    out = format_human_citation(
        "dk/lbk/2018/502/section/§6/stk.1/nr.1",
        parent_type="Provision",
        parent_name="dk/lbk/2018/502/section/§6/stk.1/nr.1",
    )
    assert out.startswith("Lbk nr. 502 af 2018")
    assert "§ 6 stk. 1 nr. 1" in out
    assert "[dk/lbk/2018/502/section/§6/stk.1/nr.1]" in out


def test_format_dk_provision_with_known_short_title():
    """When the candidate carries the human Instrument name, use it
    instead of synthesising one from doc_type."""
    out = format_human_citation(
        "dk/lbk/2018/502/section/§6/stk.1",
        parent_type="Provision",
        parent_name="Databeskyttelsesloven",
        section_path="§ 6 stk. 1",
    )
    assert "Databeskyttelsesloven" in out
    assert "§ 6 stk. 1" in out
    assert "[dk/lbk/2018/502/section/§6/stk.1]" in out


def test_format_dk_instrument():
    out = format_human_citation(
        "dk/lbk/2018/502", parent_type="Instrument", parent_name=""
    )
    assert "Lbk nr. 502 af 2018" in out
    assert "[dk/lbk/2018/502]" in out


def test_format_eu_regulation_article():
    out = format_human_citation(
        "eu/celex/32016R0679/article/art.6",
        parent_type="Provision",
        parent_name="eu/celex/32016R0679/article/art.6",
    )
    assert "Reg (EU) 2016/679" in out
    assert "Art. 6" in out
    assert "[eu/celex/32016R0679/article/art.6]" in out


def test_format_eu_directive_article():
    out = format_human_citation(
        "eu/celex/32019L0770/article/art.3",
        parent_type="Provision",
        parent_name="eu/celex/32019L0770/article/art.3",
    )
    assert "Dir (EU) 2019/770" in out
    assert "Art. 3" in out


def test_format_eu_instrument():
    out = format_human_citation(
        "eu/celex/32016R0679", parent_type="Instrument", parent_name=""
    )
    assert "Reg (EU) 2016/679" in out
    assert "[eu/celex/32016R0679]" in out


def test_format_ecli_dk_case():
    out = format_human_citation(
        "ECLI:DK:HR:2023:1234",
        parent_type="Case",
        parent_name="Forbrugersag mod Storbanken A/S",
    )
    assert "Forbrugersag mod Storbanken A/S" in out
    assert "ECLI:DK:HR:2023:1234" in out
    assert "[ECLI:DK:HR:2023:1234]" in out


def test_format_ecli_eu_case_passthrough():
    out = format_human_citation(
        "ECLI:EU:C:1993:905",
        parent_type="Case",
        parent_name="Keck Mithouard",
    )
    assert "Keck Mithouard" in out
    assert "[ECLI:EU:C:1993:905]" in out


def test_format_uk_provision_kept():
    out = format_human_citation(
        "uk/ukpga/2006/35/section/2@enacted",
        parent_type="Provision",
        parent_name="uk/ukpga/2006/35/section/2@enacted",
    )
    assert "UKPGA 2006 c.35" in out
    assert "s.2" in out
    assert "version=enacted" in out
    assert "[uk/ukpga/2006/35/section/2@enacted]" in out


def test_format_fallback_returns_bare_id():
    """Unknown id schemes must still surface the bracketed identifier."""
    out = format_human_citation(
        "cl-cluster-1234", parent_type="Case", parent_name=""
    )
    assert out == "[cl-cluster-1234]"


def test_format_case_with_name_emulates_caption():
    out = format_human_citation(
        "cl-cluster-1234",
        parent_type="Case",
        parent_name="Brown v. Board of Education",
    )
    assert "Brown v. Board of Education" in out
    assert "[cl-cluster-1234]" in out


# --- format_candidates_block ---------------------------------------------


def _candidate(parent_id: str, parent_type: str, text: str, **kw):
    from crimellm.clg.retrieval.seed import Candidate

    return Candidate(
        chunk_id="c",
        text=text,
        parent_type=parent_type,
        parent_id=parent_id,
        parent_name=kw.get("parent_name", parent_id),
        parent_jurisdiction=kw.get("parent_jurisdiction"),
        section_path=kw.get("section_path"),
        version_id=kw.get("version_id"),
        decision_date=kw.get("decision_date"),
        source="test",
        base_score=1.0,
        score=1.0,
    )


def test_candidates_block_uses_format_helpers():
    candidates = [
        _candidate(
            "eu/celex/32016R0679/article/art.6",
            "Provision",
            "Lawfulness of processing.",
        ),
        _candidate(
            "ECLI:DK:HR:2023:1234",
            "Case",
            "Højesteret afgjorde sagen.",
            parent_name="Forbrugersag",
        ),
    ]
    block = format_candidates_block(candidates, language="da")
    assert "#1 Reg (EU) 2016/679, Art. 6" in block
    assert "[eu/celex/32016R0679/article/art.6]" in block
    assert "#2 Forbrugersag" in block
    assert "[ECLI:DK:HR:2023:1234]" in block


# --- Fake synthesizer end-to-end (no LLM needed) -------------------------


def test_fake_synthesizer_uses_da_disclaimer():
    from crimellm.clg.retrieval.parse_query import parse_query
    from crimellm.clg.retrieval.synthesize import FakeSynthesizer

    q = parse_query("Hvilke regler gælder for behandling af persondata?")
    assert q.language == "da"
    candidates = [
        _candidate(
            "dk/lbk/2018/502/section/§6/stk.1",
            "Provision",
            "Behandling af personoplysninger må kun finde sted, hvis…",
            parent_name="Databeskyttelsesloven",
        )
    ]
    ans = FakeSynthesizer().synthesise(query=q, candidates=candidates, good_law={})
    assert ans.text.startswith(DISCLAIMER_DA)
    assert "Baseret på de hentede kilder" in ans.text
    # Citation guard still emits canonical id.
    assert "[dk/lbk/2018/502/section/§6/stk.1]" in ans.text


def test_fake_synthesizer_uses_en_disclaimer_for_en_query():
    from crimellm.clg.retrieval.parse_query import parse_query
    from crimellm.clg.retrieval.synthesize import FakeSynthesizer

    q = parse_query("What does Article 6 GDPR say about lawful processing?")
    assert q.language == "en"
    candidates = [
        _candidate(
            "eu/celex/32016R0679/article/art.6",
            "Provision",
            "Lawfulness of processing.",
        )
    ]
    ans = FakeSynthesizer().synthesise(query=q, candidates=candidates, good_law={})
    assert ans.text.startswith(DISCLAIMER_EN)
    assert "Based on the retrieved authorities" in ans.text


def test_empty_answer_localised_da():
    from crimellm.clg.retrieval.parse_query import parse_query
    from crimellm.clg.retrieval.synthesize import FakeSynthesizer

    q = parse_query("Hvilke regler gælder for behandling af persondata?")
    ans = FakeSynthesizer().synthesise(query=q, candidates=[], good_law={})
    assert ans.text.startswith(DISCLAIMER_DA)
    assert "Ingen kilder" in ans.text


@pytest.mark.parametrize("language", ["en", "da"])
def test_fake_synthesizer_citation_guard_still_active(language):
    """Language routing must NOT relax the strict identifier guard."""
    from crimellm.clg.retrieval.parse_query import parse_query

    q = parse_query("test").with_overrides(language=language, as_of=date(2024, 1, 1))
    cands = [_candidate("eu/celex/32016R0679", "Instrument", "GDPR text.")]
    from crimellm.clg.retrieval.synthesize import FakeSynthesizer

    ans = FakeSynthesizer().synthesise(query=q, candidates=cands, good_law={})
    # The fake synthesizer cites only ids from the retrieved set — by
    # construction no fabrication is possible.
    assert ans.citations == ["eu/celex/32016R0679"]
