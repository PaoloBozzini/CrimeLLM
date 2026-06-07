"""Parse Find Case Law (TNA) judgments — Akoma Ntoso / LegalDocML.

Each judgment becomes one ``Case`` plus a list of ``(section_ref, instrument_ref)``
references extracted from inline ``<ref>`` elements. Phase 2.3 wires these
into ``(Case)-[:INTERPRETS]->(Provision)`` edges, choosing the Provision
version that was in force on the judgment's decision date.

The XML uses the Akoma Ntoso namespace
``http://docs.oasis-open.org/legaldocml/ns/akn/3.0``.

Resolution rules for refs:
  * ``<ref href="/akn/uk/.../section/N">`` → ``(act_type, year, number, section)``
    matching ``UK_CRIMINAL_ACTS``-style triples.
  * Anything that doesn't resolve to a known statute we keep as a free-text
    reference for later analysis but do NOT emit an INTERPRETS edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from lxml import etree

from ..models import Case, Provenance

AKN_NS = {"akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}

_HREF_PATTERN = re.compile(
    r"^(?:/akn)?/uk/act/(ukpga|ukla|asp|nia|wsi)/(\d{4})/(\d+)(?:/[^/]+)?/section/([\w.]+)$"
)


@dataclass(slots=True)
class SectionRef:
    """A reference from a judgment to a statutory section.

    ``instrument_id`` matches the id minted by
    ``clg.parse.legislation_uk.instrument_id`` so the loader can resolve to
    an existing Provision without further translation.
    """

    instrument_id: str
    section_path: str
    raw_href: str


def _ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s).strip() if s else ""


def _date_or_none(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _meta_text(root: etree._Element, xpath: str) -> str:
    el = root.find(xpath, AKN_NS)
    return _ws(el.text if el is not None else "")


def _resolve_href(href: str) -> SectionRef | None:
    """Map an Akoma Ntoso URI to a SectionRef pointing at our Instrument id."""
    if not href:
        return None
    m = _HREF_PATTERN.match(href.strip())
    if not m:
        return None
    act_type, year_s, number_s, section = m.group(1), m.group(2), m.group(3), m.group(4)
    iid = f"uk/{act_type}/{int(year_s)}/{int(number_s)}"
    return SectionRef(
        instrument_id=iid,
        section_path=f"s.{section}",
        raw_href=href,
    )


def parse_judgment(
    xml_bytes: bytes,
    *,
    source_url: str | None = None,
    retrieved_at: date | None = None,
) -> tuple[Case, list[SectionRef]]:
    """Parse one Akoma Ntoso judgment XML.

    Returns ``(Case, [SectionRef])``. The Case has the neutral-citation /
    ECLI-derived id; the SectionRefs are pre-resolved to Instrument ids so
    the loader can join straight against the Provision tree.
    """
    root = etree.fromstring(xml_bytes)
    found = root.find(".//akn:judgment", AKN_NS)
    judgment = found if found is not None else root

    # FRBRWork / FRBRExpression identify the judgment.
    work_uri_el = judgment.find(".//akn:FRBRWork/akn:FRBRuri", AKN_NS)
    work_uri = work_uri_el.get("value", "") if work_uri_el is not None else ""
    # Neutral citation: prefer <neutralCitation> text; fall back to FRBRalias.
    nc_el = judgment.find(".//akn:neutralCitation", AKN_NS)
    if nc_el is None:
        nc_el = judgment.find(".//akn:FRBRalias[@name='neutralCitation']", AKN_NS)
    if nc_el is None:
        neutral_citation = ""
    elif nc_el.text:
        neutral_citation = _ws(nc_el.text)
    else:
        neutral_citation = _ws(nc_el.get("value", ""))

    # Decision date: <FRBRdate name='judgment' date='YYYY-MM-DD'/>.
    date_el = judgment.find(".//akn:FRBRWork/akn:FRBRdate", AKN_NS)
    decision_date = _date_or_none(date_el.get("date") if date_el is not None else None)

    case_name = _meta_text(judgment, ".//akn:FRBRname")
    if not case_name:
        case_name = _meta_text(judgment, ".//akn:docTitle")
    if not case_name:
        case_name = neutral_citation or work_uri

    court_id_el = judgment.find(".//akn:proprietary/akn:court", AKN_NS)
    court_id = _ws(court_id_el.text if court_id_el is not None else "")

    case_id = neutral_citation or work_uri or case_name
    case = Case(
        id=f"tna-{case_id}" if case_id else "tna-unknown",
        jurisdiction="EW",
        court_id=court_id,
        name=case_name,
        decision_date=decision_date,
        citations=[neutral_citation] if neutral_citation else [],
        provenance=[
            Provenance(
                source="find-case-law",
                source_url=source_url or work_uri,
                retrieved_at=retrieved_at or date.today(),
                source_id=case_id,
            )
        ],
    )

    refs: list[SectionRef] = []
    seen: set[tuple[str, str]] = set()
    for ref_el in judgment.iterfind(".//akn:ref", AKN_NS):
        sr = _resolve_href(ref_el.get("href", ""))
        if sr is None:
            continue
        key = (sr.instrument_id, sr.section_path)
        if key in seen:
            continue
        seen.add(key)
        refs.append(sr)

    return case, refs


def parse_judgment_file(path: Path | str, **kwargs: Any) -> tuple[Case, list[SectionRef]]:
    return parse_judgment(Path(path).read_bytes(), **kwargs)
