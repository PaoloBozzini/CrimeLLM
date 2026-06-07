"""CourtListener bulk-data downloader.

The CL bulk area exposes daily PostgreSQL CSV dumps:

    https://storage.courtlistener.com/bulk-data/<name>-YYYY-MM-DD.csv.bz2

We mirror the slim slice we need (courts, opinion-clusters, opinions,
citations) into ``data/raw/courtlistener/`` and let the parser stream them
straight from ``.bz2``. All downloads are resumable, polite, and
provenance-tagged via the existing path.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ...common.http import stream_download
from ..config import get_settings

CL_BULK_BASE = "https://storage.courtlistener.com/bulk-data"

BULK_FILES: dict[str, str] = {
    "courts": "courts-{date}.csv.bz2",
    "dockets": "dockets-{date}.csv.bz2",
    "clusters": "opinion-clusters-{date}.csv.bz2",
    "opinions": "opinions-{date}.csv.bz2",
    # The opinion->opinion citation graph (CL OpinionsCited table). This is
    # what populates `CITES` edges. CL has shipped this under several names
    # historically; `download` tries each in order.
    "citations": "citation-map-{date}.csv.bz2",
    # Reporter citations attached to clusters (e.g. "467 U.S. 837"). NOT the
    # graph edges. Used later to populate Case.citations[].
    "reporter_citations": "citations-{date}.csv.bz2",
}

# Fallback filename patterns per key, tried in order if the primary 404s.
BULK_FILES_FALLBACKS: dict[str, list[str]] = {
    "citations": [
        "citation-map-{date}.csv.bz2",
        "opinionscited-{date}.csv.bz2",
        "opinions-cited-{date}.csv.bz2",
    ],
}


def _dest_dir(custom: Path | None = None) -> Path:
    return Path(custom) if custom else get_settings().raw_root / "courtlistener"


def file_url(file_key: str, dump_date: str, base_url: str = CL_BULK_BASE) -> str:
    if file_key not in BULK_FILES:
        raise KeyError(f"unknown bulk file `{file_key}`; pick one of {list(BULK_FILES)}")
    return f"{base_url}/{BULK_FILES[file_key].format(date=dump_date)}"


def file_path(file_key: str, dump_date: str, dest_dir: Path | None = None) -> Path:
    """Return the local path for a bulk file.

    Walks the primary name + any fallbacks and returns the first existing
    file. If none exist, returns the primary path (so callers can build URLs
    or print "missing").
    """
    root = _dest_dir(dest_dir)
    names: list[str] = [BULK_FILES[file_key].format(date=dump_date)]
    for tpl in BULK_FILES_FALLBACKS.get(file_key, []):
        n = tpl.format(date=dump_date)
        if n not in names:
            names.append(n)
    for n in names:
        p = root / n
        if p.exists():
            return p
    return root / names[0]


def download(
    file_key: str,
    dump_date: str,
    *,
    dest_dir: Path | None = None,
    base_url: str = CL_BULK_BASE,
) -> Path:
    """Download a bulk file. Tries primary name then any registered fallbacks.

    On success, the local copy is named after whichever URL actually returned
    200, so subsequent `file_path(...)` calls find it.
    """
    import httpx

    candidates: list[str] = [BULK_FILES[file_key].format(date=dump_date)]
    for tpl in BULK_FILES_FALLBACKS.get(file_key, []):
        name = tpl.format(date=dump_date)
        if name not in candidates:
            candidates.append(name)

    last_err: Exception | None = None
    for name in candidates:
        url = f"{base_url}/{name}"
        dest = _dest_dir(dest_dir) / name
        try:
            return stream_download(url, dest, desc=name)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                last_err = e
                continue
            raise
    raise FileNotFoundError(
        f"none of {candidates!r} found under {base_url}/ (last error: {last_err})"
    )


def download_all(
    dump_date: str,
    *,
    files: Iterable[str] | None = None,
    dest_dir: Path | None = None,
    base_url: str = CL_BULK_BASE,
) -> dict[str, Path]:
    selected = list(files) if files is not None else list(BULK_FILES.keys())
    out: dict[str, Path] = {}
    for k in selected:
        out[k] = download(k, dump_date, dest_dir=dest_dir, base_url=base_url)
    return out
