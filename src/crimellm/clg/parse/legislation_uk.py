"""Parse legislation.gov.uk CLML XML -> ``Instrument`` + versioned ``Provision``.

The schema we parse is the Crown Legislation Markup Language (CLML) that
legislation.gov.uk serves at ``/<type>/<year>/<number>/data.xml`` and the
point-in-time variants ``/<type>/<year>/<number>/<YYYY-MM-DD>/data.xml`` or
``/<type>/<year>/<number>/enacted/data.xml``.

A single call to ``parse_act`` yields:

  * one ``Instrument`` for the Act itself (idempotent across versions),
  * one ``Provision`` per ``<P1>`` (section), tagged with the version date
    so the loader can MERGE distinct versions side-by-side in Neo4j.

Lower-level structure (subsections, paragraphs) is collapsed into the
section's ``text`` for now; Phase 3+ will revisit if the chunker wants
finer-grained granularity.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from lxml import etree

from ..models import Instrument, Provenance, Provision

LEG_UK_BASE = "https://www.legislation.gov.uk"

# CLML namespaces.
NS = {
    "leg": "http://www.legislation.gov.uk/namespaces/legislation",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ukm": "http://www.legislation.gov.uk/namespaces/metadata",
}


# --- identifiers -----------------------------------------------------------


def instrument_id(act_type: str, year: int, number: int) -> str:
    """ELI-style id (jurisdiction-prefixed for clarity inside the graph)."""
    return f"uk/{act_type}/{year}/{number}"


def provision_id(act_type: str, year: int, number: int, section: str, version: str) -> str:
    return f"{instrument_id(act_type, year, number)}/section/{section}@{version}"


# --- small helpers ---------------------------------------------------------


def _ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s).strip() if s else ""


def _date_or_none(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _itertext_clean(el: etree._Element) -> str:
    return _ws(" ".join(t for t in el.itertext() if t))


# --- metadata extraction ---------------------------------------------------


def _instrument_metadata(root: etree._Element) -> dict[str, Any]:
    title_el = root.find(".//dc:title", NS)
    title = _ws(title_el.text if title_el is not None else "")
    # ukm:Metadata sits at the top under leg:Legislation.
    enacted = root.find(".//ukm:EnactmentDate", NS)
    enacted_date = _date_or_none(enacted.get("Date") if enacted is not None else None)
    return {"title": title, "enacted_date": enacted_date}


# --- main parse ------------------------------------------------------------


def parse_act(
    xml_bytes: bytes,
    *,
    act_type: str,
    year: int,
    number: int,
    version_label: str,
    valid_from: date | None = None,
    valid_to: date | None = None,
    source_url: str | None = None,
    retrieved_at: date | None = None,
) -> tuple[Instrument, list[Provision]]:
    """Parse one whole-act CLML XML response.

    Args:
        xml_bytes: raw response body.
        act_type / year / number: act key, e.g. ("ukpga", 2006, 35) for the Fraud Act.
        version_label: free-form label for the version (e.g. "enacted",
            "current", an ISO date). Becomes part of each Provision's node id.
        valid_from / valid_to: temporal bounds for this version. If None we
            try to infer ``valid_from`` from the Act's metadata or the
            ``version_label`` (when it looks like a date).
        source_url, retrieved_at: provenance fields. ``retrieved_at`` defaults
            to today.

    Returns:
        ``(Instrument, [Provision])``. The Instrument is the same regardless
        of version (one node per Act). Provisions are per-section per-version.
    """
    root = etree.fromstring(xml_bytes)
    meta = _instrument_metadata(root)
    iid = instrument_id(act_type, year, number)

    if valid_from is None and version_label == "enacted":
        valid_from = meta.get("enacted_date") or date(year, 1, 1)
    elif valid_from is None:
        # Try to parse the label as an ISO date.
        valid_from = _date_or_none(version_label)

    prov = Provenance(
        source="legislation.gov.uk",
        source_url=source_url
        or f"{LEG_UK_BASE}/{act_type}/{year}/{number}/{version_label}/data.xml",
        retrieved_at=retrieved_at or date.today(),
        source_id=f"{iid}@{version_label}",
    )

    instrument = Instrument(
        id=iid,
        jurisdiction="UK" if act_type == "ukpga" else "EW",
        short_title=meta["title"] or f"{act_type} {year} c.{number}",
        year=year,
        provenance=[prov],
    )

    provisions: list[Provision] = []
    for p1 in root.iterfind(".//leg:Body//leg:P1", NS):
        pnum = p1.find("leg:Pnumber", NS)
        sec = _ws(pnum.text if pnum is not None else "")
        if not sec:
            continue
        parent = p1.getparent()
        heading = ""
        if parent is not None and parent.tag.endswith("}P1group"):
            t = parent.find("leg:Title", NS)
            if t is not None:
                heading = _itertext_clean(t)
        body = _itertext_clean(p1)
        if not body:
            continue
        section_path = f"s.{sec}"
        text = f"{section_path} {heading}\n\n{body}" if heading else f"{section_path}\n\n{body}"
        provisions.append(
            Provision(
                id=provision_id(act_type, year, number, sec, version_label),
                instrument_id=iid,
                jurisdiction=instrument.jurisdiction,
                section_path=section_path,
                text=text,
                valid_from=valid_from,
                valid_to=valid_to,
                version_id=version_label,
            )
        )

    return instrument, provisions


def parse_act_file(path: Path | str, **kwargs: Any) -> tuple[Instrument, list[Provision]]:
    """Convenience: read XML from disk and forward to ``parse_act``."""
    raw = Path(path).read_bytes()
    return parse_act(raw, **kwargs)
