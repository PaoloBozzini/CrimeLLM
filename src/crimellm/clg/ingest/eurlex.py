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

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ...common.http import UA, get_with_retry
from ..ingest._base import IngestContext, LoadReport, Source
from ..models import Case, Instrument, Provision
from ..parse import eurlex as P

EURLEX_BASE = P.EURLEX_BASE
CELLAR_BASE = P.CELLAR_BASE

DEFAULT_LANGUAGES: tuple[str, ...] = ("en",)
DEFAULT_FORMAT = "fmx4"  # FORMEX; AKN is "xhtml_akn" on newer endpoints.

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


def download_celex(
    client: httpx.Client,
    celex: str,
    dest_dir: Path,
    *,
    language: str = "en",
    fmt: str = DEFAULT_FORMAT,
    force: bool = False,
) -> Path | None:
    """Cache one (CELEX, language, fmt) body. Returns the path or None on 404."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = celex_path(celex, language=language, fmt=fmt, dest_dir=dest_dir)
    if out.exists() and not force:
        return out
    try:
        r = get_with_retry(client, celex_url(celex, language=language, fmt=fmt))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise
    out.write_bytes(r.content)
    return out


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
