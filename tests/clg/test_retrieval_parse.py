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
