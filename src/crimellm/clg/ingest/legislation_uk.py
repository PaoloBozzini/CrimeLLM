"""legislation.gov.uk — UK Acts downloader (point-in-time versions).

Open Government Licence v3.0. No API key needed. Polite + resumable: each
``(act, version)`` lands at a stable on-disk path and is skipped on re-run.

Versions are explicit. The default set is ``("enacted", "current")`` — two
snapshots is enough to satisfy the Phase 2 gate ("as-of two dates returns
two texts"). For deeper history pass `--versions 2010-01-01,2020-01-01,…`
or feed a list to ``LegislationUKSource``.

Whole-act XML responses contain every section valid at that version, so one
HTTP request per (act, version) is enough — about ten requests for the
default UK_CRIMINAL_ACTS list.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ...common.http import UA, get_with_retry
from ..ingest._base import IngestContext, LoadReport, Source
from ..models import Instrument, Provision
from ..parse import legislation_uk as P

LEG_UK_BASE = P.LEG_UK_BASE

# UK criminal-law staples; mirrors classifier/corpora.UK_CRIMINAL_ACTS but
# stays scoped to this module so the clg pipeline doesn't depend on the
# classifier extra.
UK_CRIMINAL_ACTS: tuple[tuple[str, int, int], ...] = (
    ("ukpga", 2006, 35),  # Fraud Act 2006
    ("ukpga", 1968, 60),  # Theft Act 1968
    ("ukpga", 1971, 38),  # Misuse of Drugs Act 1971
    ("ukpga", 1971, 48),  # Criminal Damage Act 1971
    ("ukpga", 1861, 100),  # Offences against the Person Act 1861
    ("ukpga", 1986, 64),  # Public Order Act 1986
    ("ukpga", 2003, 42),  # Sexual Offences Act 2003
    ("ukpga", 2015, 30),  # Modern Slavery Act 2015
    ("ukpga", 2010, 23),  # Bribery Act 2010
)

DEFAULT_VERSIONS = ("enacted", "current")
_VERSION_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --- URL builders ----------------------------------------------------------


def act_url(act_type: str, year: int, number: int, version: str) -> str:
    """URL for whole-act CLML XML. ``version`` is ``current``/``enacted``/ISO date."""
    if version == "current":
        return f"{LEG_UK_BASE}/{act_type}/{year}/{number}/data.xml"
    return f"{LEG_UK_BASE}/{act_type}/{year}/{number}/{version}/data.xml"


def act_path(act_type: str, year: int, number: int, version: str, dest_dir: Path) -> Path:
    """Stable on-disk location for the cached XML."""
    return dest_dir / f"{act_type}-{year}-{number}-{version}.xml"


# --- low-level fetch -------------------------------------------------------


def download_act(
    client: httpx.Client,
    act_type: str,
    year: int,
    number: int,
    version: str,
    dest_dir: Path,
    *,
    force: bool = False,
) -> Path | None:
    """Cache one (act, version) XML. Returns the local path or None on 404."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = act_path(act_type, year, number, version, dest_dir)
    if out.exists() and not force:
        return out
    try:
        r = get_with_retry(client, act_url(act_type, year, number, version))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise
    out.write_bytes(r.content)
    return out


# --- Source ABC implementation ---------------------------------------------


@dataclass
class LegislationUKSource(Source):
    name: str = "legislation_uk"
    statutes: tuple[tuple[str, int, int], ...] = UK_CRIMINAL_ACTS
    versions: tuple[str, ...] = DEFAULT_VERSIONS

    def _validate_versions(self) -> None:
        for v in self.versions:
            if v in {"enacted", "current"} or _VERSION_DATE_RE.match(v):
                continue
            raise ValueError(
                f"invalid version label {v!r}; use 'enacted', 'current', "
                "or an ISO date like '2020-01-01'"
            )

    def download(self, ctx: IngestContext) -> dict[str, Path]:
        self._validate_versions()
        dest = ctx.source_raw_dir(self.name)
        out: dict[str, Path] = {}
        with httpx.Client(headers=UA, timeout=60.0, follow_redirects=True) as client:
            for act_type, year, number in self.statutes:
                for version in self.versions:
                    p = download_act(client, act_type, year, number, version, dest)
                    if p is not None:
                        out[f"{act_type}/{year}/{number}@{version}"] = p
        return out

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:
        """Yield ``("instrument", Instrument)`` once per Act, then ``("provision", Provision)`` per (section, version)."""
        dest = ctx.source_raw_dir(self.name)
        seen_instruments: set[str] = set()
        for act_type, year, number in self.statutes:
            for version in self.versions:
                fp = act_path(act_type, year, number, version, dest)
                if not fp.exists():
                    continue
                inst, provisions = P.parse_act_file(
                    fp,
                    act_type=act_type,
                    year=year,
                    number=number,
                    version_label=version,
                )
                if inst.id not in seen_instruments:
                    yield ("instrument", inst)
                    seen_instruments.add(inst.id)
                for prov in provisions:
                    yield ("provision", prov)

    def load(self, ctx: IngestContext) -> LoadReport:
        """Push parsed Instruments + Provisions into Neo4j and return counts."""
        from ..graph.loaders import load_instruments, load_provisions

        instruments: list[Instrument] = []
        provisions: list[Provision] = []
        for kind, item in self.parse(ctx):
            if kind == "instrument":
                instruments.append(item)
            elif kind == "provision":
                provisions.append(item)

        n_inst = load_instruments(instruments, store=ctx.store)
        n_prov = load_provisions(provisions, store=ctx.store)
        return LoadReport(
            source=self.name,
            counts={"instruments": n_inst, "provisions": n_prov},
            extras={
                "statutes": len(self.statutes),
                "versions": list(self.versions),
            },
        )


# --- functional shims (mirror clg/ingest/courtlistener.py style) -----------


def download_all(
    *,
    statutes: Iterable[tuple[str, int, int]] | None = None,
    versions: Iterable[str] | None = None,
    ctx: IngestContext | None = None,
) -> dict[str, Path]:
    src = LegislationUKSource(
        statutes=tuple(statutes) if statutes is not None else UK_CRIMINAL_ACTS,
        versions=tuple(versions) if versions is not None else DEFAULT_VERSIONS,
    )
    return src.download(ctx or IngestContext())
