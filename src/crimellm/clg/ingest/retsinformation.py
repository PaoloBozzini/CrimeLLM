"""Retsinformation — accn-based ingester via the official API.

Three endpoints:

1. **Discovery / harvest** — ``https://api.retsinformation.dk/v1/Documents?date=YYYY-MM-DD``
   returns a JSON delta feed (changed-in-window) of ``Document`` records with
   ``accessionsnummer``, ``documentType.shortName``, ``href``, etc. Window
   is "last 10 days" per the published swagger; the API enforces it strictly.

2. **Slash-form → accession-number resolver** —
   ``https://www.retsinformation.dk/eli/lta/<year>/<num>.rdfa`` returns the
   SPA shell wrapped with RDFa metadata that embeds the accn (and only that
   embedding tells us the accn for any given citable slash-form ELI).
   ``resolve_accn(year, num)`` does the lookup. The ``lta`` (Lovtidende A)
   pub-media covers LOV, LBK, and BEK; the conventional ``lov`` / ``lbk``
   / ``bek`` slash-forms don't return useful RDFa.

3. **Per-document XML** — each Document carries
   ``href = "http://retsinformation.dk/eli/accn/<ACCN>/xml"`` pointing at
   the structured **LexDania** XML body.

Operator workflow:

  a. *Discover* recent changes via ``--discover [--since YYYY-MM-DD]``, or
     *resolve* a citable slash-form via ``--resolve lov/2018/502``, or just
     paste a slash-form into ``--items`` (resolver runs transparently).
  b. *Ingest* the chosen accns/slash-forms via ``--items <CSV>``.

Open data, free reuse with attribution (Civilstyrelsen). Polite +
resumable.
"""

from __future__ import annotations

import json as _json
import re
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


def rdfa_url(year: int, num: int, *, pub_media: str = "lta") -> str:
    """RDFa resolver URL for ``year/num`` under a publication-media class.

    ``lta`` (Lovtidende A) covers LOV, LBK, and BEK. ``ltc`` (Lovtidende C)
    is rarely needed. The conventional ``lov`` / ``lbk`` / ``bek`` slugs
    don't return useful RDFa — Retsinformation only embeds the accn for
    the publication-media routing path.
    """
    return f"https://www.retsinformation.dk/eli/{pub_media}/{year}/{num}.rdfa"


# An accession number is the literal id Retsinformation's XML uses.
# Format: <prefix><year><7 digits>, prefix ∈ {A, B, C, D}.
_ACCN_RE = re.compile(r"^[ABCD]20\d{9}$")
_SLASH_RE = re.compile(r"^(?P<doc>[a-z]+)/(?P<year>\d{4})/(?P<num>\d+)$")
_RDFA_ACCN_RE = re.compile(rb'"([ABCD]20\d{9})"')


def is_accn(value: str) -> bool:
    return bool(_ACCN_RE.match(value.strip()))


def is_slash_form(value: str) -> bool:
    return bool(_SLASH_RE.match(value.strip()))


def resolve_accn(
    year: int,
    num: int,
    *,
    pub_media: str = "lta",
    client: httpx.Client | None = None,
) -> str | None:
    """Look up the accession number for a slash-form ELI.

    Fetches ``rdfa_url(year, num, pub_media=...)`` and pulls the first
    ``"A20XXXXXXXXX"`` / ``"B…"`` / ``"C…"`` literal out of the RDFa
    payload. Returns ``None`` when the slug isn't recognised (the SPA
    returns its 2.8 KB shell with no embedded RDFa).
    """
    url = rdfa_url(year, num, pub_media=pub_media)
    owns_client = client is None
    if client is None:
        client = httpx.Client(headers=UA, timeout=30.0, follow_redirects=True)
    try:
        try:
            r = get_with_retry(client, url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        # Empty RDFa shell ≈ 2.8 KB; real RDFa payload is ≥ ~5 KB. Bail
        # early so we don't false-positive on `<meta property="og:..."` etc.
        if len(r.content) < 4_000:
            return None
        m = _RDFA_ACCN_RE.search(r.content)
        return m.group(1).decode("ascii") if m else None
    finally:
        if owns_client:
            client.close()


def normalise_items(
    items: Iterable[str],
    *,
    pub_media: str = "lta",
    client: httpx.Client | None = None,
) -> list[tuple[str, tuple[str, int, int] | None]]:
    """Map a mixed list of accn / slash-form entries to ``(accn, triple_or_None)``.

    * ``"A20180050230"`` → ``("A20180050230", None)``.
    * ``"lov/2018/502"`` → looks up the accn via ``resolve_accn`` →
      ``("A20180050230", ("lov", 2018, 502))``.

    Slash-form entries that fail to resolve raise ``ValueError`` so the
    operator knows which entry is the problem; the alternative (silent
    skip) hides typos that look like working downloads.
    """
    out: list[tuple[str, tuple[str, int, int] | None]] = []
    owns_client = client is None
    if client is None:
        client = httpx.Client(headers=UA, timeout=30.0, follow_redirects=True)
    try:
        for raw in items:
            entry = raw.strip()
            if not entry:
                continue
            if is_accn(entry):
                out.append((entry, None))
                continue
            m = _SLASH_RE.match(entry)
            if not m:
                raise ValueError(
                    f"unrecognised --items entry {entry!r}: "
                    "expected accn (e.g. A20180050230) or slash-form "
                    "(e.g. lov/2018/502)"
                )
            doc_type = m["doc"]
            year = int(m["year"])
            num = int(m["num"])
            accn = resolve_accn(year, num, pub_media=pub_media, client=client)
            if accn is None:
                raise ValueError(
                    f"could not resolve {entry!r} to an accession number via "
                    f"{rdfa_url(year, num, pub_media=pub_media)} — does the "
                    "document exist? Try --discover or browse the SPA."
                )
            out.append((accn, (doc_type, year, num)))
    finally:
        if owns_client:
            client.close()
    return out


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
