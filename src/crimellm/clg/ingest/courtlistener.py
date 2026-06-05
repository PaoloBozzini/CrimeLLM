"""CourtListener bulk-data downloader.

The CL bulk area exposes daily PostgreSQL CSV dumps:

    https://storage.courtlistener.com/bulk-data/<name>-YYYY-MM-DD.csv.bz2

We mirror the slim slice we need (courts, opinion-clusters, opinions,
citations) into ``data/raw/courtlistener/`` and let the parser stream them
straight from ``.bz2``. All downloads are resumable, polite, and
provenance-tagged via the existing path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..._http import stream_download
from ..config import get_settings

CL_BULK_BASE = "https://storage.courtlistener.com/bulk-data"

BULK_FILES: dict[str, str] = {
    "courts":     "courts-{date}.csv.bz2",
    "dockets":    "dockets-{date}.csv.bz2",
    "clusters":   "opinion-clusters-{date}.csv.bz2",
    "opinions":   "opinions-{date}.csv.bz2",
    "citations":  "citations-{date}.csv.bz2",
}


def _dest_dir(custom: Path | None = None) -> Path:
    return Path(custom) if custom else get_settings().raw_root / "courtlistener"


def file_url(file_key: str, dump_date: str, base_url: str = CL_BULK_BASE) -> str:
    if file_key not in BULK_FILES:
        raise KeyError(f"unknown bulk file `{file_key}`; pick one of {list(BULK_FILES)}")
    return f"{base_url}/{BULK_FILES[file_key].format(date=dump_date)}"


def file_path(file_key: str, dump_date: str, dest_dir: Path | None = None) -> Path:
    return _dest_dir(dest_dir) / BULK_FILES[file_key].format(date=dump_date)


def download(
    file_key: str,
    dump_date: str,
    *,
    dest_dir: Path | None = None,
    base_url: str = CL_BULK_BASE,
) -> Path:
    url = file_url(file_key, dump_date, base_url)
    dest = file_path(file_key, dump_date, dest_dir)
    return stream_download(url, dest, desc=dest.name)


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
