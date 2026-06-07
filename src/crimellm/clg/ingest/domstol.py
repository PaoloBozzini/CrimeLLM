"""domstol.dk — Danish court judgment downloader (free corpus).

The official portal publishes Højesteret + landsret judgments mostly as
PDFs (some HTML). There's no clean direct-by-ECLI URL scheme like
EUR-Lex, so this ingester takes an **operator-curated list** of
``(ecli, url, court_id)`` triples. Discovery (find all 2024 Højesteret
judgments) is deferred — operators usually work from a firm matter
index or a published case list.

Open access, free reuse with attribution. Commercial reporter coverage
(Karnov / Ufr) lives in ``ingest/karnov.py`` and is licence-gated.

Lifecycle mirrors the other sources:
* ``download`` — per-ECLI PDF / HTML cache on disk; skipped on re-run.
* ``parse`` — yields ``("court", Court)`` for the seed hierarchy then
  ``("case", Case)`` per judgment.
* ``load`` — MERGE Courts first, then Cases (so the DECIDED edge fires).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ...common.http import UA, get_with_retry
from ..ingest._base import IngestContext, LoadReport, Source
from ..models import Case, Court
from ..parse import domstol as P


# --- Court seeds ----------------------------------------------------------


# Court hierarchy: higher level = more senior. Højesteret binds landsret
# binds byret in DK civil-law style (persuasive weight, not strict
# common-law binding — see analysis doc §2). The level numbers mirror the
# UK courts' convention in the existing UK ingester.
DK_COURTS: tuple[Court, ...] = (
    Court(id="hr", jurisdiction="DK", name="Højesteret", level=3, parent_id=None),
    Court(id="olr", jurisdiction="DK", name="Østre Landsret", level=2, parent_id="hr"),
    Court(id="vlr", jurisdiction="DK", name="Vestre Landsret", level=2, parent_id="hr"),
    Court(id="byret", jurisdiction="DK", name="Byret (generic)", level=1, parent_id="olr"),
)


# --- URL builders ----------------------------------------------------------


def cache_path(ecli: str, dest_dir: Path, *, suffix: str = ".pdf") -> Path:
    """Stable per-ECLI cache filename. Replace ECLI separators for the FS."""
    safe = ecli.replace(":", "_").replace(".", "_")
    return dest_dir / f"{safe}{suffix}"


# --- low-level fetch ------------------------------------------------------


def download_judgment(
    client: httpx.Client,
    ecli: str,
    url: str,
    dest_dir: Path,
    *,
    suffix: str | None = None,
    force: bool = False,
) -> Path | None:
    """Cache one judgment body. Returns the path or None on 404.

    ``suffix`` defaults to ``.pdf`` for ``.pdf`` URLs, else ``.html``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if suffix is None:
        suffix = ".pdf" if url.lower().endswith(".pdf") else ".html"
    out = cache_path(ecli, dest_dir, suffix=suffix)
    if out.exists() and not force:
        return out
    try:
        r = get_with_retry(client, url)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise
    out.write_bytes(r.content)
    return out


# --- Source ABC implementation -------------------------------------------


@dataclass
class JudgmentRef:
    """Operator-supplied work item for one judgment."""

    ecli: str
    url: str
    court_id: str | None = None  # inferred from ECLI when None
    name: str | None = None
    decision_date: str | None = None  # ISO; parser fills from body when missing


@dataclass
class DomstolSource(Source):
    """Pull a list of DK judgments by (ECLI, URL).

    ``items`` is the work queue. Operator usually starts from a firm
    matter index; SPARQL-style discovery against domstol.dk is deferred.
    """

    name: str = "domstol"
    items: tuple[JudgmentRef, ...] = field(default_factory=tuple)

    def download(self, ctx: IngestContext) -> dict[str, Path]:
        if not self.items:
            return {}
        dest = ctx.source_raw_dir(self.name)
        out: dict[str, Path] = {}
        with httpx.Client(headers=UA, timeout=120.0, follow_redirects=True) as client:
            for ref in self.items:
                p = download_judgment(client, ref.ecli, ref.url, dest)
                if p is not None:
                    out[ref.ecli] = p
        return out

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:
        """Yield ``("court", Court)`` seeds then ``("case", Case)`` rows."""
        # Seed courts once per run — load_courts is idempotent.
        for court in DK_COURTS:
            yield ("court", court)

        dest = ctx.source_raw_dir(self.name)
        for ref in self.items:
            # Look for either cached suffix.
            fp_pdf = cache_path(ref.ecli, dest, suffix=".pdf")
            fp_html = cache_path(ref.ecli, dest, suffix=".html")
            fp = fp_pdf if fp_pdf.exists() else (fp_html if fp_html.exists() else None)
            if fp is None:
                continue
            pr = P.parse_judgment_file(
                fp,
                ecli=ref.ecli,
                court_id=ref.court_id,
                name=ref.name,
                source_url=ref.url,
            )
            yield ("case", pr.case)
            # citation_hits passed through ctx.params so downstream link
            # phase can pick them up without re-parsing — keeps Phase 5
            # citation-edge emission consistent with the link/cite_registry
            # contract.
            ctx.params.setdefault("citation_hits", []).extend(pr.citation_hits)

    def load(self, ctx: IngestContext) -> LoadReport:
        from ..graph.loaders import load_cases, load_courts

        courts: list[Court] = []
        cases: list[Case] = []
        for kind, item in self.parse(ctx):
            if kind == "court":
                courts.append(item)
            elif kind == "case":
                cases.append(item)

        n_courts = load_courts(courts, store=ctx.store)
        n_cases = load_cases(cases, store=ctx.store)
        return LoadReport(
            source=self.name,
            counts={
                "courts": n_courts,
                "cases": n_cases,
                "citation_hits": len(ctx.params.get("citation_hits", [])),
            },
            extras={"items": len(self.items)},
        )


# --- functional shim ------------------------------------------------------


def download_all(
    items: Iterable[JudgmentRef],
    *,
    ctx: IngestContext | None = None,
) -> dict[str, Path]:
    src = DomstolSource(items=tuple(items))
    return src.download(ctx or IngestContext())
