"""Shared HTTP utilities. Used by `crimellm.corpora` and `crimellm.clg.ingest.*`.

Single source of truth for User-Agent, retry/backoff (Retry-After-aware),
JSONL writing, and streaming downloads with resume.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

UA = {"User-Agent": "crimellm/0.1 (+https://github.com/)"}


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    max_retries: int = 4,
    timeout: float = 60.0,
) -> httpx.Response:
    """GET with exponential backoff on 429 (honours `Retry-After`)."""
    for attempt in range(max_retries + 1):
        r = client.get(url, params=params, timeout=timeout)
        if r.status_code == 429 and attempt < max_retries:
            wait = float(r.headers.get("Retry-After") or 2**attempt)
            tqdm.write(f"[http] 429 on {url} -- sleep {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def stream_download(
    url: str,
    dest: str | Path,
    *,
    headers: dict[str, str] | None = None,
    chunk: int = 1 << 20,
    resume: bool = True,
    desc: str | None = None,
) -> Path:
    """Resumable streaming download.

    Skips entirely if `dest` exists and the server returns a matching
    `Content-Length` (no partial range needed); otherwise uses a Range request
    when the server advertises `Accept-Ranges: bytes`.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = dict(UA)
    if headers:
        h.update(headers)

    with httpx.Client(headers=h, timeout=300.0, follow_redirects=True) as client:
        if resume and dest.exists():
            # HEAD to check size; if equal, treat as complete.
            try:
                head = client.head(url)
                head.raise_for_status()
                total = int(head.headers.get("Content-Length", 0))
                if total and dest.stat().st_size == total:
                    return dest
            except httpx.HTTPError:
                pass  # fall through to GET

        start = dest.stat().st_size if (resume and dest.exists()) else 0
        req_headers = dict(h)
        if start:
            req_headers["Range"] = f"bytes={start}-"

        with client.stream("GET", url, headers=req_headers) as r:
            if r.status_code == 416:  # range not satisfiable -> already complete
                return dest
            r.raise_for_status()
            mode = "ab" if start and r.status_code == 206 else "wb"
            total = int(r.headers.get("Content-Length", 0)) + (start if mode == "ab" else 0)
            with (
                open(dest, mode) as f,
                tqdm(
                    total=total or None,
                    initial=start if mode == "ab" else 0,
                    unit="B",
                    unit_scale=True,
                    desc=desc or dest.name,
                    leave=False,
                ) as bar,
            ):
                for block in r.iter_bytes(chunk_size=chunk):
                    f.write(block)
                    bar.update(len(block))
    return dest
