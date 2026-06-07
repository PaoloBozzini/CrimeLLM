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
