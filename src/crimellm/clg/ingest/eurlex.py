"""EUR-Lex / CELLAR — direct-by-CELEX downloader.

Phase 3 uses the per-CELEX content URLs that EUR-Lex serves under
``publications.europa.eu/resource/celex/<CELEX>``. A query parameter
selects format and language; we fetch Akoma Ntoso 3.0 when available and
fall back to FORMEX. SPARQL discovery (find all directives published
since ``$date``) is deferred to Phase 3.5 — most operator workflows
start from a known CELEX list or a hand-picked spreadsheet.

Polite + resumable, mirrors ``ingest/legislation_uk.py``:

* one (CELEX, language, format) → one cached file on disk
* skipped on re-run unless ``--force``
* retry / timeout via ``crimellm.common.http``

Per the analysis brief (§3, §6), licence is reuse-with-attribution and no
API key is needed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ...common.http import UA
from ..ingest._base import IngestContext, LoadReport, Source
from ..models import Case, Instrument, Provision
from ..parse import eurlex as P

EURLEX_BASE = P.EURLEX_BASE
CELLAR_BASE = P.CELLAR_BASE

DEFAULT_LANGUAGES: tuple[str, ...] = ("en",)
DEFAULT_FORMAT = "fmx4"  # FORMEX — the only XML serialisation CELLAR serves reliably.

# CELLAR content negotiation needs ISO 639-2/T (3-letter). The codebase uses
# 2-letter ISO 639-1 elsewhere; map at the edge.
_LANG3 = {"en": "eng", "da": "dan", "de": "deu", "fr": "fra"}

# Format → CELLAR Accept MIME. CELLAR rejects ``application/xml;type=akn``
# with HTTP 400; AKN is only available as hand-curated fixtures. FORMEX
# (``fmx4``) covers both legislation (<ACT>) and CJEU judgments (<JUDGMENT>).
_FMT_ACCEPT = {
    "fmx4": "application/xml;type=fmx4",
}

# Multi-choice (HTTP 300) HTML response lists per-manifestation DOC_N URLs.
# When multiple are returned, the body usually sits at the highest DOC_N
# (DOC_1 is a per-issue ToC; DOC_2 is the actual document body).
_DOC_HREF_RE = re.compile(
    r'href="(?P<url>http://publications\.europa\.eu/resource/cellar/'
    r'[a-f0-9-]+\.\d+\.\d+/DOC_(?P<n>\d+))"'
)

# Languages the firm actually reads. Operator picks via --lang CSV at the
# CLI; this list documents the intent.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "da", "de", "fr")


# --- URL builders ----------------------------------------------------------


def celex_url(celex: str, *, language: str = "en", fmt: str = DEFAULT_FORMAT) -> str:
    """Resource URL for the CELEX in the given language + format."""
    return f"{CELLAR_BASE}/{celex}?language={language}&format={fmt}"


def celex_path(
    celex: str,
    *,
    language: str,
    fmt: str,
    dest_dir: Path,
) -> Path:
    """Stable on-disk location for the cached body."""
    return dest_dir / f"{celex}.{language}.{fmt}.xml"


# --- low-level fetch -------------------------------------------------------


def _pick_body_url(multi_choice_html: str) -> str | None:
    """Pick the manifestation body URL from a CELLAR 300 Multiple-Choice page.

    The HTML lists each DOC_N as ``<a href="...DOC_N">``. DOC_1 is usually a
    per-issue ToC and DOC_2 (when present) is the document body. Prefer the
    highest N; fall back to the only one when N=1.
    """
    matches = list(_DOC_HREF_RE.finditer(multi_choice_html))
    if not matches:
        return None
    matches.sort(key=lambda m: int(m.group("n")))
    return matches[-1].group("url")


def download_celex(
    client: httpx.Client,
    celex: str,
    dest_dir: Path,
    *,
    language: str = "en",
    fmt: str = DEFAULT_FORMAT,
    force: bool = False,
) -> Path | None:
    """Cache one (CELEX, language, fmt) body. Returns the path or None on 404.

    Two-step CELLAR content negotiation:

    1. GET ``resource/celex/<CELEX>`` with ``Accept: application/xml;type=fmx4``
       and ``Accept-Language: <ISO-639-2/T>``. CELLAR resolves the work to the
       right manifestation and returns either the body (200) or a Multiple-Choice
       page listing the per-stream DOC_N URLs (300).
    2. On 300, follow the highest DOC_N href to fetch the actual body bytes.

    The legacy ``?language=&format=`` query-param URL is **not** honoured —
    CELLAR ignores those and serves the NOTICE metadata wrapper instead, which
    is useless to the body parser.
    """
    if fmt not in _FMT_ACCEPT:
        raise ValueError(
            f"unsupported format {fmt!r}; CELLAR only serves "
            f"{sorted(_FMT_ACCEPT)} via content negotiation"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = celex_path(celex, language=language, fmt=fmt, dest_dir=dest_dir)
    if out.exists() and not force:
        return out

    lang3 = _LANG3.get(language.lower(), language.lower())
    accept = _FMT_ACCEPT[fmt]
    headers = {**UA, "Accept": accept, "Accept-Language": lang3}
    url = f"{CELLAR_BASE}/{celex}"

    try:
        r = client.get(url, headers=headers, timeout=60.0, follow_redirects=True)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise

    if r.status_code == 404:
        return None
    if r.status_code == 200:
        out.write_bytes(r.content)
        return out
    if r.status_code == 300:
        body_url = _pick_body_url(r.text)
        if body_url is None:
            return None
        r2 = client.get(body_url, headers={**UA}, timeout=60.0, follow_redirects=True)
        if r2.status_code == 404:
            return None
        r2.raise_for_status()
        out.write_bytes(r2.content)
        return out
    r.raise_for_status()
    return None


# --- Source ABC implementation --------------------------------------------


@dataclass
class EurLexSource(Source):
    """Pull a list of CELEX ids in one or more languages.

    ``celex_ids`` is the work queue. Operator usually starts from a known
    bundle (GDPR + e-commerce + consumer-rights core) and grows it as
    matters land. SPARQL-driven discovery is future work.
    """

    name: str = "eurlex"
    celex_ids: tuple[str, ...] = field(default_factory=tuple)
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    fmt: str = DEFAULT_FORMAT

    def download(self, ctx: IngestContext) -> dict[str, Path]:
        if not self.celex_ids:
            return {}
        dest = ctx.source_raw_dir(self.name)
        out: dict[str, Path] = {}
        with httpx.Client(headers=UA, timeout=60.0, follow_redirects=True) as client:
            for celex in self.celex_ids:
                for lang in self.languages:
                    p = download_celex(client, celex, dest, language=lang, fmt=self.fmt)
                    if p is not None:
                        out[f"{celex}@{lang}.{self.fmt}"] = p
        return out

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:
        """Yield mixed-type model rows.

        ``("instrument", Instrument)`` and ``("provision", Provision)`` for
        legislation; ``("case", Case)`` for judgments; ``("implements",
        (src_id, tgt_id))`` for IMPLEMENTS-seed pairs the regulation parser
        recovered from preamble CELEX mentions.
        """
        dest = ctx.source_raw_dir(self.name)
        seen_instruments: set[str] = set()
        seen_cases: set[str] = set()
        for celex in self.celex_ids:
            kind = P.celex_kind(celex)
            for lang in self.languages:
                fp = celex_path(celex, language=lang, fmt=self.fmt, dest_dir=dest)
                if not fp.exists():
                    continue
                if kind == "legislation":
                    pr = P.parse_regulation_file(fp, celex=celex, language=lang)
                    if pr.instrument.id not in seen_instruments:
                        yield ("instrument", pr.instrument)
                        seen_instruments.add(pr.instrument.id)
                    for prov in pr.provisions:
                        yield ("provision", prov)
                    for cited in pr.cites_celex:
                        yield (
                            "implements",
                            (
                                pr.instrument.id,
                                P.instrument_id_from_celex(cited),
                                cited,
                            ),
                        )
                elif kind == "case":
                    jp = P.parse_judgment_file(fp, celex=celex, language=lang)
                    if jp.case.id not in seen_cases:
                        yield ("case", jp.case)
                        seen_cases.add(jp.case.id)
                else:
                    # Skip treaties / international agreements for now.
                    continue

    # --- autofetch single-ID fetch (Phase C.2) -----------------------------

    def supports_single_fetch(self) -> bool:
        return True

    def fetch_one(
        self,
        ctx: IngestContext,
        cite_id: str,
        *,
        client: httpx.Client | None = None,
    ) -> dict[str, Path]:
        """Download one EUR-Lex resource by CELEX / ECLI:EU / ELI slash-form.

        All three shapes are normalised to a CELEX before hitting the cellar
        endpoint, which is the only single-doc retrieval the public API
        supports cleanly. Default language + format come from the instance
        (``languages[0]``, ``fmt``); the autofetch worker doesn't override
        either today.
        """
        from ..autofetch.exceptions import UnsupportedCite

        celex = _to_celex(cite_id)
        if celex is None:
            raise UnsupportedCite(
                f"cite {cite_id!r}: not a CELEX, ECLI:EU, or supported ELI slash-form."
            )

        language = self.languages[0] if self.languages else "en"
        dest = ctx.source_raw_dir(self.name)
        owns_client = client is None
        if client is None:
            client = httpx.Client(headers=UA, timeout=60.0, follow_redirects=True)
        try:
            path = download_celex(client, celex, dest, language=language, fmt=self.fmt)
        finally:
            if owns_client:
                client.close()
        if path is None:
            raise UnsupportedCite(
                f"cite {cite_id!r}: CELEX {celex} returned 404 from CELLAR."
            )
        return {cite_id: path}

    def load(self, ctx: IngestContext) -> LoadReport:
        from ..graph.loaders import (
            load_cases,
            load_implements,
            load_instruments,
            load_provisions,
        )

        instruments: list[Instrument] = []
        provisions: list[Provision] = []
        cases: list[Case] = []
        implements: list[tuple[str, str, str]] = []
        for kind, item in self.parse(ctx):
            if kind == "instrument":
                instruments.append(item)
            elif kind == "provision":
                provisions.append(item)
            elif kind == "case":
                cases.append(item)
            elif kind == "implements":
                implements.append(item)

        n_inst = load_instruments(instruments, store=ctx.store)
        n_prov = load_provisions(provisions, store=ctx.store)
        n_case = load_cases(cases, store=ctx.store)
        n_imp = load_implements(implements, store=ctx.store)
        return LoadReport(
            source=self.name,
            counts={
                "instruments": n_inst,
                "provisions": n_prov,
                "cases": n_case,
                "implements": n_imp,
            },
            extras={
                "celex_ids": len(self.celex_ids),
                "languages": list(self.languages),
                "format": self.fmt,
            },
        )


# --- autofetch helpers ----------------------------------------------------
#
# Conversion rules (single source of truth, kept module-local so the rest of
# the pipeline keeps using its own canonical IDs):
#
# - CELEX shape ``[1-9]\d{4}[A-Z]{1,2}\d{4}`` → returned verbatim.
# - ECLI:EU:<C|T|F>:<year>:<num> → CELEX ``6<year><CJ|TJ|FJ><num zero-padded 4>``.
# - eu/<reg|dir|dec>/<year>/<num>[/...] → CELEX ``3<year><R|L|D><num zero-padded 4>``.

import re as _re

_CELEX_RE_EXACT = _re.compile(r"^[1-9]\d{4}[A-Z]{1,2}\d{4}$")
_ECLI_EU_RE = _re.compile(r"^ECLI:EU:(?P<court>[CTF]):(?P<year>\d{4}):(?P<num>\d+)$")
_ELI_EU_RE = _re.compile(
    r"^eu/(?P<type>reg|dir|dec)/(?P<year>\d{4})/(?P<num>\d+)(?:/.*)?$"
)
_ECLI_COURT_TO_CELEX = {"C": "CJ", "T": "TJ", "F": "FJ"}
_ELI_TYPE_TO_CELEX = {"reg": "R", "dir": "L", "dec": "D"}


def _to_celex(cite_id: str) -> str | None:
    """Normalise CELEX / ECLI:EU / ELI to a CELEX. ``None`` when unrecognised."""
    if _CELEX_RE_EXACT.match(cite_id):
        return cite_id
    m = _ECLI_EU_RE.match(cite_id)
    if m:
        court = _ECLI_COURT_TO_CELEX[m["court"]]
        return f"6{m['year']}{court}{int(m['num']):04d}"
    m = _ELI_EU_RE.match(cite_id)
    if m:
        type_letter = _ELI_TYPE_TO_CELEX[m["type"]]
        return f"3{m['year']}{type_letter}{int(m['num']):04d}"
    return None


# --- functional shim (mirror legislation_uk.download_all) -----------------


def download_all(
    celex_ids: Iterable[str],
    *,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
    fmt: str = DEFAULT_FORMAT,
    ctx: IngestContext | None = None,
) -> dict[str, Path]:
    src = EurLexSource(
        celex_ids=tuple(celex_ids),
        languages=tuple(languages),
        fmt=fmt,
    )
    return src.download(ctx or IngestContext())
