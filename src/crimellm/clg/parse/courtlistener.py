"""Stream CourtListener bulk CSVs -> internal `models` dataclasses.

Designed for resumable, memory-light passes over very large dumps. All readers
accept `.csv` or `.csv.bz2` transparently. Column lookup is by name (CL has
shifted column orders before).

FK chain we walk for Phase 1:
    opinion -> opinion_cluster (cluster_id)
    opinion_cluster -> docket  (docket_id)
    docket -> court            (court_id)
    citation: opinion_id -> opinion_id        (resolved to cluster_id pairs)
"""

from __future__ import annotations

import bz2
import csv
import io
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date as _date
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from ..models import Case, Citation, Court, Provenance

# CL bulk rows carry multi-MB text fields (syllabus, headnotes, html_*).
# The stdlib default of 131072 chars rejects them outright.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# Prefer pbzip2 / lbzip2 (parallel) -> bzip2 (system, C, fast) -> Python bz2.
_BZ2_BIN = shutil.which("pbzip2") or shutil.which("lbzip2") or shutil.which("bzip2")

_TODAY = _date.today()
_PROV_SRC = "courtlistener-bulk"
_PROV_BASE = "https://storage.courtlistener.com/bulk-data"


@contextmanager
def _open_csv(path: Path):
    """Stream a CL bulk CSV (plain or .bz2) as a text iterator.

    For .bz2 we shell out to bzip2 -dc when available — orders of magnitude
    faster than Python's stdlib bz2 on multi-GB dumps. Falls back gracefully.
    """
    p = Path(path)
    if p.suffix == ".bz2" and _BZ2_BIN:
        proc = subprocess.Popen(
            [_BZ2_BIN, "-dc", str(p)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1 << 22,
        )
        wrapper = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
        try:
            yield wrapper
        finally:
            try:
                wrapper.close()
            except Exception:
                pass
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        return

    if p.suffix == ".bz2":
        with bz2.open(p, "rt", newline="", encoding="utf-8", errors="replace") as fh:
            yield fh
    else:
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            yield fh


def _row_iter(
    path: Path,
    *,
    desc: str | None = None,
    progress: bool = False,
) -> Iterator[dict[str, str]]:
    with _open_csv(path) as fh:
        reader = csv.DictReader(fh)
        if not progress:
            for row in reader:
                yield row
            return
        bar = tqdm(reader, desc=desc or Path(path).name, unit="row", unit_scale=True)
        for row in bar:
            yield row


def _indexed_row_iter(
    path: Path,
    columns: list[str | tuple[str, ...]],
    *,
    desc: str | None = None,
    progress: bool = True,
) -> Iterator[tuple[str, ...]]:
    """Fast path: csv.reader + column indices. Skips dict construction.

    Each entry in ``columns`` is either a single column name or a tuple of
    synonyms; the first synonym present in the header wins. CL renames
    columns between dumps (citing_opinion_id ↔ citing_opinion ↔ id_from),
    so callers pass tuples to stay tolerant.

    ~3-5x faster than DictReader on rows with many large fields.
    """
    with _open_csv(path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return
        idx: list[int] = []
        for c in columns:
            choices = (c,) if isinstance(c, str) else tuple(c)
            picked = next((header.index(x) for x in choices if x in header), None)
            if picked is None:
                raise ValueError(
                    f"missing column in {path.name}: none of {choices!r} in header={header!r}"
                )
            idx.append(picked)
        max_idx = max(idx)
        if progress:
            reader = tqdm(reader, desc=desc or Path(path).name, unit="row", unit_scale=True)
        for row in reader:
            if len(row) <= max_idx:
                continue
            yield tuple(row[i] for i in idx)


def _date_or_none(s: str | None) -> _date | None:
    if not s:
        return None
    s = s.split(" ")[0].split("T")[0]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _prov(source_id: str, source_url: str | None = None) -> Provenance:
    return Provenance(
        source=_PROV_SRC,
        source_url=source_url or f"https://www.courtlistener.com/?q=cl:{source_id}",
        retrieved_at=_TODAY,
        source_id=source_id,
    )


# --- courts ----------------------------------------------------------------


def iter_courts(courts_csv: Path, *, progress: bool = False) -> Iterator[Court]:
    for row in _row_iter(courts_csv, desc="courts", progress=progress):
        cid = (row.get("id") or "").strip()
        if not cid:
            continue
        try:
            level = int(row.get("position") or 0)
        except ValueError:
            level = 0
        yield Court(
            id=cid,
            jurisdiction="US",
            name=(row.get("full_name") or row.get("short_name") or cid).strip(),
            level=level,
            parent_id=(row.get("parent_court_id") or None) or None,
        )


# --- dockets (cluster_id -> court_id resolution) ---------------------------


def build_docket_to_court(
    dockets_csv: Path | None,
    allowed_docket_ids: Iterable[str] | None = None,
    *,
    progress: bool = False,
) -> dict[str, str]:
    """Stream dockets.csv into a slim `{docket_id: court_id}` map."""
    if dockets_csv is None:
        return {}
    allow = set(allowed_docket_ids) if allowed_docket_ids is not None else None
    out: dict[str, str] = {}
    for did, cid in _indexed_row_iter(
        Path(dockets_csv),
        ["id", "court_id"],
        desc="dockets",
        progress=progress,
    ):
        did = did.strip()
        cid = cid.strip()
        if not did or not cid:
            continue
        if allow is not None and did not in allow:
            continue
        out[did] = cid
    return out


# --- clusters -> Case -------------------------------------------------------


def iter_cases(
    clusters_csv: Path,
    *,
    docket_to_court: dict[str, str] | None = None,
    limit: int | None = None,
    progress: bool = False,
) -> Iterator[Case]:
    """Yield one `Case` per opinion cluster. Cluster id is the Case id."""
    docket_to_court = docket_to_court or {}
    n = 0
    for row in _row_iter(clusters_csv, desc="clusters", progress=progress):
        cid = (row.get("id") or "").strip()
        if not cid:
            continue
        docket_id = (row.get("docket_id") or "").strip()
        court_id = docket_to_court.get(docket_id, "")
        cluster_url = f"https://www.courtlistener.com/opinion/{cid}/"
        yield Case(
            id=f"cl-cluster-{cid}",
            jurisdiction="US",
            court_id=court_id,
            name=(
                row.get("case_name")
                or row.get("case_name_short")
                or row.get("case_name_full")
                or ""
            ).strip(),
            decision_date=_date_or_none(row.get("date_filed")),
            citations=[],
            provenance=[_prov(cid, cluster_url)],
        )
        n += 1
        if limit and n >= limit:
            return


def cluster_ids(
    clusters_csv: Path,
    *,
    limit: int | None = None,
    progress: bool = False,
) -> set[str]:
    """Lightweight pre-pass: collect cluster ids (optionally first `limit`)."""
    out: set[str] = set()
    for (cid,) in _indexed_row_iter(
        Path(clusters_csv),
        ["id"],
        desc="clusters:ids",
        progress=progress,
    ):
        cid = cid.strip()
        if cid:
            out.add(cid)
            if limit and len(out) >= limit:
                return out
    return out


def docket_ids_for_clusters(
    clusters_csv: Path,
    allowed_clusters: set[str],
    *,
    progress: bool = False,
) -> set[str]:
    out: set[str] = set()
    for cid, did in _indexed_row_iter(
        Path(clusters_csv),
        ["id", "docket_id"],
        desc="clusters:dockets",
        progress=progress,
    ):
        if cid.strip() in allowed_clusters:
            did = did.strip()
            if did:
                out.add(did)
    return out


# --- opinions (opinion_id -> cluster_id) -----------------------------------


def build_opinion_to_cluster(
    opinions_csv: Path,
    *,
    allowed_clusters: set[str] | None = None,
    progress: bool = False,
) -> dict[str, str]:
    """Stream opinions.csv -> `{opinion_id: cluster_id}` map.

    The slow pass. Prefer `build_opinion_cluster_index` once and read the
    sidecar via `load_opinion_cluster_index` for repeat use.
    """
    out: dict[str, str] = {}
    for oid, cl in _indexed_row_iter(
        Path(opinions_csv),
        ["id", "cluster_id"],
        desc="opinions",
        progress=progress,
    ):
        oid = oid.strip()
        cl = cl.strip()
        if not oid or not cl:
            continue
        if allowed_clusters is not None and cl not in allowed_clusters:
            continue
        out[oid] = cl
    return out


def opinion_cluster_index_path(opinions_csv: Path) -> Path:
    """Sidecar location for the cached `(opinion_id, cluster_id)` index.

    Example: opinions-2024-12-31.csv.bz2 -> opinions-2024-12-31.opcluster.csv
    """
    p = Path(opinions_csv)
    stem = p.name
    if stem.endswith(".csv.bz2"):
        stem = stem[: -len(".csv.bz2")]
    elif stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    return p.parent / f"{stem}.opcluster.csv"


def build_opinion_cluster_index(
    opinions_csv: Path,
    *,
    dest: Path | None = None,
    progress: bool = True,
) -> Path:
    """One-shot pass that writes a slim 2-col CSV: `opinion_id,cluster_id`.

    Run this once per dump. Subsequent loads load the sidecar in seconds.
    """
    src = Path(opinions_csv)
    out = Path(dest) if dest else opinion_cluster_index_path(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["opinion_id", "cluster_id"])
        for oid, cl in _indexed_row_iter(
            src,
            ["id", "cluster_id"],
            desc="indexing opinions",
            progress=progress,
        ):
            oid = oid.strip()
            cl = cl.strip()
            if oid and cl:
                w.writerow([oid, cl])
    tmp.replace(out)
    return out


def load_opinion_cluster_index(
    idx_csv: Path,
    *,
    allowed_clusters: set[str] | None = None,
    progress: bool = False,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for oid, cl in _indexed_row_iter(
        Path(idx_csv),
        ["opinion_id", "cluster_id"],
        desc="opcluster",
        progress=progress,
    ):
        if allowed_clusters is not None and cl not in allowed_clusters:
            continue
        out[oid] = cl
    return out


# --- citations -------------------------------------------------------------


def iter_citations(
    citations_csv: Path,
    opinion_to_cluster: dict[str, str],
    *,
    progress: bool = False,
) -> Iterator[Citation]:
    """Yield `Case -[:CITES]-> Case` edges, resolved via opinion->cluster.

    Drops self-cites and rows where either side is outside the cluster scope.
    Treatment stays `neutral` here; Phase 5 fills it.
    """
    for citing_op, cited_op, depth_s in _indexed_row_iter(
        Path(citations_csv),
        [
            ("citing_opinion_id", "citing_opinion", "id_from", "from_id", "citing"),
            ("cited_opinion_id", "cited_opinion", "id_to", "to_id", "cited"),
            ("depth", "weight", "n"),
        ],
        desc="citations",
        progress=progress,
    ):
        citing = opinion_to_cluster.get(citing_op.strip())
        cited = opinion_to_cluster.get(cited_op.strip())
        if not citing or not cited or citing == cited:
            continue
        try:
            depth = float(depth_s) if depth_s else 1.0
        except ValueError:
            depth = 1.0
        yield Citation(
            citing_case_id=f"cl-cluster-{citing}",
            cited_case_id=f"cl-cluster-{cited}",
            treatment="neutral",
            citing_sentence="",
            weight=depth,
        )
