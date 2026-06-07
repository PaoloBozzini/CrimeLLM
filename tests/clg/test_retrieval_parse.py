"""Query parser + small helpers — no Neo4j needed."""

from __future__ import annotations

from datetime import date

from crimellm.clg.retrieval.parse_query import parse_query


def test_parse_query_defaults_to_today_no_jurisdiction() -> None:
    q = parse_query("what does the law say about robbery?")
    assert q.jurisdiction is None
    assert q.as_of == date.today()


def test_parse_query_infers_uk() -> None:
    q = parse_query("Does s.2 of the Fraud Act 2006 cover phishing?")
    assert q.jurisdiction == "UK"


def test_parse_query_infers_us() -> None:
    q = parse_query("What is 18 U.S.C. § 1341 about?")
    assert q.jurisdiction == "US"


def test_parse_query_extracts_iso_date() -> None:
    q = parse_query("Sentencing for fraud as of 2018-05-12, UK?")
    assert q.as_of == date(2018, 5, 12)
    assert q.jurisdiction == "UK"


def test_with_overrides_replaces_fields() -> None:
    q = parse_query("anything")
    q2 = q.with_overrides(jurisdiction="US", as_of="2020-01-01")
    assert q2.jurisdiction == "US"
    assert q2.as_of == date(2020, 1, 1)
    # Original is untouched (dataclass slots).
    assert q.jurisdiction is None


def test_with_overrides_accepts_date_obj() -> None:
    q = parse_query("anything").with_overrides(as_of=date(2010, 6, 1))
    assert q.as_of == date(2010, 6, 1)


# --- DK + EU cue inference (Phase 4.5 / T7.1) -----------------------------


def test_parse_query_infers_dk_named_statute() -> None:
    q = parse_query("Hvad indebærer straffelovens § 279 om bedrageri?")
    assert q.jurisdiction == "DK"


def test_parse_query_infers_dk_court_tier() -> None:
    q = parse_query("Har Højesteret afgjort om aftalelovens § 36 i forbrugersager?")
    assert q.jurisdiction == "DK"


def test_parse_query_infers_dk_ecli() -> None:
    q = parse_query("Hvordan fortolkes ECLI:DK:HR:2023:123?")
    assert q.jurisdiction == "DK"


def test_parse_query_infers_eu_treaty() -> None:
    q = parse_query("How does Article 101 TFEU apply to vertical agreements?")
    assert q.jurisdiction == "EU"


def test_parse_query_infers_eu_gdpr() -> None:
    q = parse_query("Has the CJEU interpreted GDPR Article 6(1)(f) on marketing?")
    assert q.jurisdiction == "EU"


def test_parse_query_infers_eu_danish_phrasing() -> None:
    # Danish caller asking about an EU instrument — EU cues outscore DK.
    q = parse_query("Hvordan har EU-Kommissionen og Rådet anvendt forordning 2016/679?")
    assert q.jurisdiction == "EU"


def test_parse_query_ties_return_none() -> None:
    # Equal hits on DK + EU → no bias.
    q = parse_query("straffelovens § 279 og TFEU artikel 101")
    assert q.jurisdiction is None


def test_parse_query_no_cues_returns_none() -> None:
    q = parse_query("a generic question about something legal")
    assert q.jurisdiction is None


# --- Phase 7: language detection (T7.2) -----------------------------------


def test_detect_language_diacritics_lock_in_da() -> None:
    from crimellm.clg.retrieval.parse_query import detect_language

    lang, conf = detect_language("Højesteret har truffet en afgørelse.")
    assert lang == "da"
    assert conf > 0.3


def test_detect_language_en_pure() -> None:
    from crimellm.clg.retrieval.parse_query import detect_language

    lang, conf = detect_language("What does the Fraud Act 2006 say about phishing?")
    assert lang == "en"
    assert conf > 0.0


def test_detect_language_da_without_diacritics() -> None:
    """Real-world: short DA query with no æ/ø/å — stopwords + bigrams +
    suffixes must still carry."""
    from crimellm.clg.retrieval.parse_query import detect_language

    lang, _ = detect_language(
        "Hvilke regler gælder for behandling af persondata efter loven?"
    )
    # 'æ' appears in 'gælder' — still DA; check fully ASCII version too.
    assert lang == "da"

    lang2, _ = detect_language(
        "Hvilke regler galder for behandling af persondata efter loven?"
    )
    # Stripped diacritic: stopwords (hvilke/regler/for/af/efter) + DA suffixes
    # (-en, -er) still tip the scale.
    assert lang2 == "da"


def test_detect_language_short_input_falls_back_to_en() -> None:
    from crimellm.clg.retrieval.parse_query import detect_language

    assert detect_language("")[0] == "en"
    assert detect_language("hi")[0] == "en"


def test_detect_language_mixed_en_with_da_term() -> None:
    """EN query mentioning a DA statute name should stay EN — the bulk of
    the sentence is EN function words."""
    from crimellm.clg.retrieval.parse_query import detect_language

    lang, _ = detect_language(
        "What does the Danish statute straffelovens § 279 say about fraud?"
    )
    assert lang == "en"


def test_parse_query_carries_language_into_query() -> None:
    q = parse_query("Højesteret har truffet en afgørelse om aftalelovens § 36.")
    assert q.language == "da"
    assert q.language_confidence > 0.0


def test_parse_query_default_language_en() -> None:
    q = parse_query("Does the Fraud Act 2006 cover phishing?")
    assert q.language == "en"


def test_with_overrides_language() -> None:
    q = parse_query("English query body here.").with_overrides(language="da")
    assert q.language == "da"


# --- Phase 7: enabled-jurisdiction filter (T7.3) --------------------------


def _settings_with_enabled(codes: list[str]):
    from crimellm.clg.config import Settings

    return Settings(enabled_jurisdictions=codes)


def test_parse_query_clears_disabled_jurisdiction() -> None:
    """With DK + EU only, a US-flavoured query has its inferred
    jurisdiction stripped to None."""
    s = _settings_with_enabled(["DK", "EU"])
    q = parse_query("What is 18 U.S.C. § 1341 about?", settings=s)
    assert q.jurisdiction is None  # US not in enabled set


def test_parse_query_keeps_enabled_jurisdiction() -> None:
    s = _settings_with_enabled(["DK", "EU"])
    q = parse_query("Hvad indebærer straffelovens § 279 om bedrageri?", settings=s)
    assert q.jurisdiction == "DK"


def test_parse_query_override_bypasses_enabled_filter() -> None:
    """CLI ``--jurisdiction DK`` (via with_overrides) should win even when
    DK is disabled — caller-knows-best."""
    s = _settings_with_enabled(["US", "UK"])
    q = parse_query("a generic query", settings=s).with_overrides(jurisdiction="DK")
    assert q.jurisdiction == "DK"


def test_parse_query_no_inferred_no_filter() -> None:
    """When no cues match, the enabled filter is a no-op (result already
    None)."""
    s = _settings_with_enabled(["US"])
    q = parse_query("entirely generic prose with no cues", settings=s)
    assert q.jurisdiction is None
