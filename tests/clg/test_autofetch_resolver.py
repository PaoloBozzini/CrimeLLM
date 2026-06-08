"""Phase B.3: cite_id → source name dispatch.

The resolver is intentionally narrow: input is a canonical ID produced by the
``cite_registry`` parsers, output is the source-module name responsible for
fetching it (or ``None`` when no parser handles that shape — the worker logs
and skips). Per-source ``fetch_one`` knows how to parse its own IDs.
"""

from __future__ import annotations

import pytest

from crimellm.clg.autofetch.resolver import resolve


@pytest.mark.parametrize(
    "cite_id,expected_source",
    [
        # DK statutes via Retsinformation ELI.
        ("eli/lov/2020/171", "retsinformation"),
        ("eli/lbk/2018/1156", "retsinformation"),
        # DK statute slug shape from cite_dk normalisation.
        ("DK/straffeloven/section/279", "retsinformation"),
        ("DK/straffeloven/section/279/stk/2", "retsinformation"),
        # EU case law via ECLI:EU.
        ("ECLI:EU:C:2014:317", "eurlex"),
        ("ECLI:EU:T:2020:1", "eurlex"),
        # EU legislation via CELEX (sector 3 = legislation).
        ("32016R0679", "eurlex"),
        ("32019L0770", "eurlex"),
        # EU case law via CELEX (sector 6).
        ("61991CJ0267", "eurlex"),
        # EU ELI.
        ("eu/reg/2016/679", "eurlex"),
        # UK legislation ELI.
        ("uk/ukpga/2018/12", "legislation_uk"),
        # US opinion id.
        ("courtlistener:opinion:12345", "courtlistener"),
    ],
)
def test_resolve_known_shapes(cite_id: str, expected_source: str) -> None:
    assert resolve(cite_id) == expected_source


@pytest.mark.parametrize(
    "cite_id",
    [
        # DK case-law via ECLI — domstol scrape not in v1 per design doc.
        "ECLI:DK:HR:2023:123",
        # DK Ufr — Karnov subscription required, not in v1 dispatch.
        "U.2010.1234.H",
        # Unrecognised garbage.
        "totally-not-a-citation",
        "",
        # UK neutral cite — find_case_law mapping deferred.
        "[2019] EWCA Civ 12",
    ],
)
def test_resolve_unknown_returns_none(cite_id: str) -> None:
    assert resolve(cite_id) is None
