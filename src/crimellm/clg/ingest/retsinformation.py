"""Retsinformation — direct-by-ELI downloader for Danish primary law.

Pulls love / lovbekendtgørelser / bekendtgørelser from the official
Retsinformation portal under the ELI URL scheme:

    https://www.retsinformation.dk/eli/<doc_type>/<year>/<num>

Open data, free reuse with attribution (Civilstyrelsen). Polite +
resumable, mirrors ``ingest/legislation_uk.py`` and ``ingest/eurlex.py``:

* one (doc_type, year, num) → one cached XML on disk
* skipped on re-run unless ``--force``
* shared ``crimellm.common.http.get_with_retry`` for retry/timeout

SPARQL-style discovery (find all lbk's in 2024) is deferred; operators
typically work from a known list (firm matter index → ELI list).
"""

from __future__ import annotations

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


# --- URL builders ----------------------------------------------------------


def eli_xml_url(doc_type: str, year: int, num: int) -> str:
    """Retsinformation serves XML when the ``Accept: application/xml`` header
    is sent; the public URL is the ELI path itself. Some operators prefer
    the ``?format=xml`` query when their HTTP client can't set headers
    cleanly — we use the path form and rely on the Accept header.
    """
    return f"{RETSINFO_BASE}/eli/{doc_type.lower()}/{year}/{num}"


def eli_path(doc_type: str, year: int, num: int, dest_dir: Path) -> Path:
    return dest_dir / f"{doc_type.lower()}-{year}-{num}.xml"


# --- low-level fetch ------------------------------------------------------


_XML_HEADERS: dict[str, str] = dict(UA, **{"Accept": "application/xml"})


def download_eli(
    client: httpx.Client,
    doc_type: str,
    year: int,
    num: int,
    dest_dir: Path,
    *,
    force: bool = False,
) -> Path | None:
    """Cache one (doc_type, year, num) XML. Returns the path or None on 404."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = eli_path(doc_type, year, num, dest_dir)
    if out.exists() and not force:
        return out
    try:
        r = get_with_retry(client, eli_xml_url(doc_type, year, num))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise
    out.write_bytes(r.content)
    return out


# --- Source ABC implementation --------------------------------------------


@dataclass
class RetsinformationSource(Source):
    """Pull a list of DK statutes by (doc_type, year, num).

    ``items`` is the work queue. Operator usually starts from a known
    bundle (e.g. databeskyttelsesloven + aftaleloven + straffeloven core)
    and extends it as matters land.
    """

    name: str = "retsinformation"
    items: tuple[tuple[str, int, int], ...] = field(default_factory=tuple)
    explode_subparagraphs: bool = True

    def download(self, ctx: IngestContext) -> dict[str, Path]:
        if not self.items:
            return {}
        dest = ctx.source_raw_dir(self.name)
        out: dict[str, Path] = {}
        with httpx.Client(
            headers=_XML_HEADERS, timeout=60.0, follow_redirects=True
        ) as client:
            for doc_type, year, num in self.items:
                p = download_eli(client, doc_type, year, num, dest)
                if p is not None:
                    out[f"{doc_type}/{year}/{num}"] = p
        return out

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:
        """Yield mixed-type model rows.

        ``("instrument", Instrument)``, ``("provision", Provision)``, and
        ``("implements", (dk_instrument_id, eu_celex_instrument_id, raw_celex))``
        for each EU directive / regulation the DK preamble cites.
        """
        dest = ctx.source_raw_dir(self.name)
        seen_instruments: set[str] = set()
        for doc_type, year, num in self.items:
            fp = eli_path(doc_type, year, num, dest)
            if not fp.exists():
                continue
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
                "items": len(self.items),
                "explode_subparagraphs": self.explode_subparagraphs,
            },
        )


# --- functional shim ------------------------------------------------------


def download_all(
    items: Iterable[tuple[str, int, int]],
    *,
    ctx: IngestContext | None = None,
) -> dict[str, Path]:
    src = RetsinformationSource(items=tuple(items))
    return src.download(ctx or IngestContext())
