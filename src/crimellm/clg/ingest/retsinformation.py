"""Retsinformation — accn-based ingester via the official API.

Two endpoints:

1. **Discovery / harvest** — ``https://api.retsinformation.dk/v1/Documents?date=YYYY-MM-DD``
   returns a JSON delta feed (changed-in-window) of ``Document`` records with
   ``accessionsnummer``, ``documentType.shortName``, ``href``, etc. Window
   is "last 10 days" per the published swagger; the API enforces it strictly.

2. **Per-document XML** — each Document carries
   ``href = "http://retsinformation.dk/eli/accn/<ACCN>/xml"`` pointing at
   the structured **LexDania** XML body.

Operator workflow (Phase 14.5):

  a. *Discover* a set of `Document` records via ``--since`` (harvest)
     or by hand (browse the SPA and copy the accession number from the URL).
  b. *Ingest* the chosen accns via ``--items A20180050229,B20260050805``.

The accession number is the API's stable identifier; the slash-form
``lbk/2018/502`` is the *citable* identifier but not directly mappable
to an accn (it's a publication-date-encoded internal id). Resolver
deferred to a future phase that mirrors the full Retsinformation
catalog via the daily harvest feed.

Open data, free reuse with attribution (Civilstyrelsen). Polite +
resumable.
"""

from __future__ import annotations

import json as _json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ...common.http import UA, get_with_retry
from ..ingest._base import IngestContext, LoadReport, Source
from ..models import Instrument, Provision
from ..parse import retsinformation as P

RETSINFO_BASE = P.RETSINFO_BASE
RETSINFO_API_BASE = "https://api.retsinformation.dk"
RETSINFO_XML_BASE = "http://retsinformation.dk/eli/accn"


# --- URL builders ---------------------------------------------------------


def accn_xml_url(accn: str) -> str:
    """Direct LexDania XML download for an accession number."""
    return f"{RETSINFO_XML_BASE}/{accn.strip()}/xml"


def discover_url(date_iso: str | None = None) -> str:
    """Daily-delta harvest endpoint. ``date_iso`` must be within the last
    10 days per the API contract; omit to use today."""
    if date_iso:
        return f"{RETSINFO_API_BASE}/v1/Documents?date={date_iso}"
    return f"{RETSINFO_API_BASE}/v1/Documents"


def accn_cache_path(accn: str, dest_dir: Path) -> Path:
    """Stable on-disk filename keyed by accession number."""
    return dest_dir / f"{accn.strip()}.xml"


# --- low-level fetch ------------------------------------------------------


def download_accn(
    client: httpx.Client,
    accn: str,
    dest_dir: Path,
    *,
    force: bool = False,
) -> Path | None:
    """Cache one accession's LexDania XML. Returns the path or None on 404."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = accn_cache_path(accn, dest_dir)
    if out.exists() and not force:
        return out
    try:
        r = get_with_retry(client, accn_xml_url(accn))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise
    out.write_bytes(r.content)
    return out


def discover_documents(
    client: httpx.Client,
    date_iso: str | None = None,
) -> list[dict[str, Any]]:
    """Harvest the daily-delta feed. Returns list of ``Document`` dicts.

    Each item carries ``accessionsnummer``, ``documentId``, ``documentType``,
    ``changeDate``, ``reasonForChange``, and the direct-XML ``href``.
    """
    r = get_with_retry(client, discover_url(date_iso))
    data = r.json()
    return list(data) if isinstance(data, list) else []


# --- Source ABC implementation -------------------------------------------


@dataclass
class RetsinformationSource(Source):
    """Pull a list of DK statutes by accession number.

    ``accns`` is the work queue. ``slash_form_map`` is optional —
    when the operator has both the accn AND the slash-form id available
    (e.g. via the SPA URL), populate ``slash_form_map[accn] = (doc_type,
    year, num)`` so the parser can build the canonical Instrument id.
    Otherwise the ingester derives ``(doc_type, year, num)`` from the
    LexDania ``<Meta>`` block (less reliable).
    """

    name: str = "retsinformation"
    accns: tuple[str, ...] = field(default_factory=tuple)
    slash_form_map: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    explode_subparagraphs: bool = True

    def download(self, ctx: IngestContext) -> dict[str, Path]:
        if not self.accns:
            return {}
        dest = ctx.source_raw_dir(self.name)
        out: dict[str, Path] = {}
        with httpx.Client(headers=UA, timeout=60.0, follow_redirects=True) as client:
            for accn in self.accns:
                p = download_accn(client, accn, dest)
                if p is not None:
                    out[accn] = p
        return out

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:
        """Yield mixed-type model rows.

        ``("instrument", Instrument)``, ``("provision", Provision)``, and
        ``("implements", (dk_instrument_id, eu_celex_instrument_id, raw_celex))``
        for each EU CELEX the LexDania ``<EuReferences>`` block (or
        preamble fallback) lists.
        """
        dest = ctx.source_raw_dir(self.name)
        seen_instruments: set[str] = set()
        for accn in self.accns:
            fp = accn_cache_path(accn, dest)
            if not fp.exists():
                continue

            # Resolve (doc_type, year, num) — operator hint or LexDania <Meta>.
            triple = self.slash_form_map.get(accn)
            if triple is None:
                triple = _infer_slash_form(fp)
                if triple is None:
                    # Fall back to a synthesised slug; better than crashing.
                    triple = ("lbk", 0, 0)

            doc_type, year, num = triple
            pr = P.parse_statute_file(
                fp,
                doc_type=doc_type,
                year=year,
                num=num,
                explode_subparagraphs=self.explode_subparagraphs,
            )
            if pr.instrument.id not in seen_instruments:
                yield ("instrument", pr.instrument)
                seen_instruments.add(pr.instrument.id)
            for prov in pr.provisions:
                yield ("provision", prov)
            for celex in pr.cites_eu_celex:
                yield (
                    "implements",
                    (pr.instrument.id, f"eu/celex/{celex}", celex),
                )

    def load(self, ctx: IngestContext) -> LoadReport:
        from ..graph.loaders import (
            load_implements,
            load_instruments,
            load_provisions,
        )

        instruments: list[Instrument] = []
        provisions: list[Provision] = []
        implements: list[tuple[str, str, str]] = []
        for kind, item in self.parse(ctx):
            if kind == "instrument":
                instruments.append(item)
            elif kind == "provision":
                provisions.append(item)
            elif kind == "implements":
                implements.append(item)

        n_inst = load_instruments(instruments, store=ctx.store)
        n_prov = load_provisions(provisions, store=ctx.store)
        n_imp = load_implements(implements, store=ctx.store)
        return LoadReport(
            source=self.name,
            counts={
                "instruments": n_inst,
                "provisions": n_prov,
                "implements": n_imp,
            },
            extras={
                "accns": len(self.accns),
                "explode_subparagraphs": self.explode_subparagraphs,
            },
        )


# --- helpers --------------------------------------------------------------


def _infer_slash_form(xml_path: Path) -> tuple[str, int, int] | None:
    """Read the LexDania ``<Meta>`` block to pull ``(doc_type, year, num)``.

    Best-effort: maps ``DocumentType.shortName`` like ``"LBK H"`` →
    ``"lbk"``, then takes ``<Year>`` and ``<Number>``. Returns ``None``
    when the schema doesn't match (e.g. legacy synthetic fixture without
    these elements).
    """
    from lxml import etree

    try:
        root = etree.parse(str(xml_path)).getroot()
    except Exception:  # noqa: BLE001
        return None
    if root.tag != "Dokument":
        return None
    meta = root.find("Meta")
    if meta is None:
        return None

    def _text(tag: str) -> str:
        el = meta.find(tag)
        return (el.text or "").strip() if el is not None and el.text else ""

    dt_raw = _text("DocumentType")  # e.g. "LBK H#LOKDOK03"
    year_raw = _text("Year")
    num_raw = _text("Number")
    if not (dt_raw and year_raw and num_raw):
        return None
    try:
        year = int(year_raw)
        num = int(num_raw)
    except ValueError:
        return None

    code = dt_raw.split("#", 1)[0].strip().split()[0].lower()  # "lbk", "bek", etc.
    return (code, year, num)


# --- functional shims ----------------------------------------------------


def download_all(
    accns: Iterable[str],
    *,
    ctx: IngestContext | None = None,
) -> dict[str, Path]:
    src = RetsinformationSource(accns=tuple(accns))
    return src.download(ctx or IngestContext())


def discover_all(
    date_iso: str | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper: open an HTTP client and harvest the daily feed."""
    with httpx.Client(headers=UA, timeout=60.0, follow_redirects=True) as client:
        return discover_documents(client, date_iso=date_iso)
