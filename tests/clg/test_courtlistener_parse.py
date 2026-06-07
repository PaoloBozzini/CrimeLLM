from __future__ import annotations

from pathlib import Path

from crimellm.clg.parse import courtlistener as P

FIX = Path(__file__).parent / "fixtures" / "courtlistener"
DATE = "2024-01-01"


def _f(name: str) -> Path:
    return FIX / f"{name}-{DATE}.csv"


def test_iter_courts() -> None:
    courts = list(P.iter_courts(_f("courts")))
    ids = {c.id for c in courts}
    assert {"scotus", "ca9", "nysd", "cand"} == ids
    scotus = next(c for c in courts if c.id == "scotus")
    assert scotus.jurisdiction == "US"
    assert scotus.level == 1
    assert "Supreme Court" in scotus.name


def test_docket_to_court_full() -> None:
    m = P.build_docket_to_court(_f("dockets"))
    assert m["1001"] == "scotus"
    assert m["1005"] == "nysd"
    assert len(m) == 6


def test_docket_to_court_scoped() -> None:
    m = P.build_docket_to_court(_f("dockets"), allowed_docket_ids={"1001", "1004"})
    assert m == {"1001": "scotus", "1004": "ca9"}


def test_iter_cases_with_court_resolution() -> None:
    docket_to_court = P.build_docket_to_court(_f("dockets"))
    cases = list(P.iter_cases(_f("opinion-clusters"), docket_to_court=docket_to_court))
    assert len(cases) == 6
    chevron = next(c for c in cases if c.id == "cl-cluster-2001")
    assert chevron.court_id == "scotus"
    assert "Chevron" in chevron.name
    assert chevron.decision_date is not None and chevron.decision_date.year == 1984
    assert chevron.provenance and chevron.provenance[0].source == "courtlistener-bulk"


def test_iter_cases_respects_limit() -> None:
    cases = list(P.iter_cases(_f("opinion-clusters"), limit=2))
    assert len(cases) == 2
    assert cases[0].id == "cl-cluster-2001"


def test_opinion_to_cluster_filtering() -> None:
    full = P.build_opinion_to_cluster(_f("opinions"))
    assert full == {
        "3001": "2001",
        "3002": "2002",
        "3003": "2003",
        "3004": "2004",
        "3005": "2005",
        "3006": "2006",
    }
    scoped = P.build_opinion_to_cluster(_f("opinions"), allowed_clusters={"2001", "2002"})
    assert scoped == {"3001": "2001", "3002": "2002"}


def test_iter_citations_resolves_to_clusters() -> None:
    op = P.build_opinion_to_cluster(_f("opinions"))
    cites = list(P.iter_citations(_f("citations"), op))
    # 8 rows in fixture; none are self-cites.
    assert len(cites) == 8
    # Inbound to Chevron (cl-cluster-2001) should be 5.
    chevron_in = [c for c in cites if c.cited_case_id == "cl-cluster-2001"]
    assert len(chevron_in) == 5
    assert all(c.treatment == "neutral" for c in cites)
    assert all(c.weight >= 1.0 for c in cites)
