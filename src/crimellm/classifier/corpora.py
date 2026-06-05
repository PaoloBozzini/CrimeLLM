"""Legal-corpus ingestion -> JSONL.

Each record:
  {id, text, source, citation, type: "statute"|"judgment", metadata}

Sources:
  - US Code via govinfo.gov (API or direct content URLs).
  - US Code via local USLM XML (manual download from uscode.house.gov).
  - CourtListener REST v4 (/search/?type=o -> /opinions/{id}/ two-hop).
  - UK legislation.gov.uk (Acts of Parliament, whole-act XML).

CLI:
    python -m crimellm.corpora us-code        --out data/corpora/usc18 [--titles 18] [--max-sections N]
    python -m crimellm.corpora courtlistener  --out data/corpora/cl    [--max-docs N]
    python -m crimellm.corpora uk             --out data/corpora/uk    [--statutes "ukpga/2006/35" "ukpga/1968/60"]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from ..common.http import UA
from ..common.http import get_with_retry as _get_with_retry
from ..common.http import write_jsonl as _write_jsonl_shared
from ..env import load_env

GOVINFO_API = "https://api.govinfo.gov"
GOVINFO_CONTENT = "https://www.govinfo.gov/content/pkg"
COURTLISTENER_API = "https://www.courtlistener.com/api/rest/v4"
LEG_UK = "https://www.legislation.gov.uk"

LEG_UK_NS = {
    "leg": "http://www.legislation.gov.uk/namespaces/legislation",
    "dc": "http://purl.org/dc/elements/1.1/",
}

# Default UK criminal-law statutes (type, year, number).
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


# --- shared utilities -------------------------------------------------------


def _ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s).strip() if s else ""


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    _write_jsonl_shared(records, path)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL corpus file produced by the ingestion helpers."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- US Code: USLM XML parser -----------------------------------------------


def _section_text(section) -> str:
    parts: list[str] = []
    for el in section.iter():
        if el.text:
            parts.append(el.text)
        if el.tail:
            parts.append(el.tail)
    return _ws(" ".join(parts))


def parse_us_code(xml_path: str | Path, title_hint: str | None = None) -> Iterator[dict[str, Any]]:
    """Parse USLM XML (file or directory), yielding one record per <section>."""
    from lxml import etree

    xml_path = Path(xml_path)
    files = sorted(xml_path.glob("*.xml")) if xml_path.is_dir() else [xml_path]

    def _local(el, name):
        return el.tag.endswith("}" + name) or el.tag == name

    for f in files:
        root = etree.parse(str(f)).getroot()

        title_num = title_hint
        if not title_num:
            for el in root.iter():
                if _local(el, "title"):
                    num_el = next((c for c in el.iter() if _local(c, "num")), None)
                    if num_el is not None:
                        m = re.search(r"\d+", (num_el.text or num_el.get("value", "") or ""))
                        if m:
                            title_num = m.group(0)
                            break
        title_num = title_num or "?"

        for section in (el for el in root.iter() if _local(el, "section")):
            num_el = next((c for c in section.iter() if _local(c, "num")), None)
            heading_el = next((c for c in section.iter() if _local(c, "heading")), None)
            num = _ws(num_el.text if num_el is not None else "").lstrip("§").strip().rstrip(".")
            heading = _ws(heading_el.text if heading_el is not None else "")
            body = _section_text(section)
            if not body or not num:
                continue
            citation = f"{title_num} U.S.C. § {num}"
            yield {
                "id": f"usc-{title_num}-{num}",
                "text": f"{citation}. {heading}\n\n{body}" if heading else f"{citation}\n\n{body}",
                "source": "us_code",
                "citation": citation,
                "type": "statute",
                "metadata": {"title": title_num, "section": num, "heading": heading},
            }


# --- US Code: govinfo HTML --------------------------------------------------


class _USCSectionHTMLParser(HTMLParser):
    """Extract the section heading + statutory body from govinfo USC HTML."""

    def __init__(self):
        super().__init__()
        self._buf: list[str] = []
        self._h3_buf: list[str] = []
        self._keep_depth = 0
        self._in_h3 = False
        self.heading: str = ""

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "")
        if tag == "h3" and "section-head" in cls:
            self._in_h3 = True
        elif tag == "p" and cls.startswith("statutory-body"):
            self._keep_depth += 1
            if self._buf and not self._buf[-1].endswith("\n"):
                self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag == "h3" and self._in_h3:
            self._in_h3 = False
            self.heading = _ws("".join(self._h3_buf))
        elif tag == "p" and self._keep_depth:
            self._keep_depth -= 1
            self._buf.append("\n")

    def handle_data(self, data):
        if self._in_h3:
            self._h3_buf.append(data)
        elif self._keep_depth:
            self._buf.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._buf)).strip()


def _parse_usc_html(html: str) -> tuple[str, str]:
    p = _USCSectionHTMLParser()
    p.feed(html)
    return p.heading, p.text()


def _usc_record(granule_id: str, title: str, year: int, heading: str, body: str) -> dict[str, Any]:
    m = re.search(r"sec([\w.]+)$", granule_id)
    sec = m.group(1) if m else granule_id
    citation = f"{title} U.S.C. § {sec}"
    return {
        "id": granule_id,
        "text": f"{citation}. {heading}\n\n{body}",
        "source": "us_code",
        "citation": citation,
        "type": "statute",
        "metadata": {
            "title": title,
            "section": sec,
            "heading": heading,
            "package_id": f"USCODE-{year}-title{title}",
            "year": year,
        },
    }


def _iter_govinfo_granules(
    package_id: str, client: httpx.Client, api_key: str
) -> Iterator[dict[str, Any]]:
    url = f"{GOVINFO_API}/packages/{package_id}/granules"
    params: dict | None = {"pageSize": 100, "offsetMark": "*", "api_key": api_key}
    while url:
        data = _get_with_retry(client, url, params=params).json()
        for g in data.get("granules", []):
            if g.get("granuleClass") == "LEAF":
                yield g
        url = data.get("nextPage")
        params = {"api_key": api_key} if url else None


def fetch_us_code_sections(
    base_path: str | Path,
    granule_ids: list[str],
    year: int = 2023,
    title: str = "18",
) -> int:
    """Fetch hand-picked USC sections via direct www.govinfo.gov URLs (no API key)."""
    package_id = f"USCODE-{year}-title{title}"
    records: list[dict[str, Any]] = []
    with httpx.Client(headers=UA) as client:
        for gid in tqdm(granule_ids, desc=package_id):
            try:
                r = _get_with_retry(client, f"{GOVINFO_CONTENT}/{package_id}/html/{gid}.htm")
            except httpx.HTTPStatusError as e:
                print(f"[corpora] skip {gid}: {e.response.status_code}")
                continue
            heading, body = _parse_usc_html(r.text)
            if body:
                records.append(_usc_record(gid, title, year, heading, body))
    out = Path(str(base_path) + ".jsonl")
    _write_jsonl(records, out)
    print(f"[corpora] wrote {len(records)} records -> {out}")
    return len(records)


def download_us_code(
    base_path: str | Path,
    titles: Iterable[str] = ("18",),
    year: int = 2023,
    api_key: str | None = None,
    max_sections: int | None = None,
    xml_path: str | Path | None = None,
) -> int:
    """Walk all sections of selected USC titles via the govinfo granule API.

    `api_key` falls back to env `GOVINFO_API_KEY`, then `"DEMO_KEY"` (30/hour).
    Pass `xml_path` to skip the API and parse a local USLM XML download.
    """
    if api_key is None:
        load_env()
        api_key = os.environ.get("GOVINFO_API_KEY") or "DEMO_KEY"
    out = Path(str(base_path) + ".jsonl")

    if xml_path is not None:
        records = list(parse_us_code(xml_path))
    else:
        records = []
        with httpx.Client(headers=UA) as client:
            for title in titles:
                package_id = f"USCODE-{year}-title{title}"
                granules = list(_iter_govinfo_granules(package_id, client, api_key))
                if max_sections is not None:
                    remaining = max_sections - len(records)
                    if remaining <= 0:
                        break
                    granules = granules[:remaining]
                for g in tqdm(granules, desc=package_id):
                    try:
                        r = _get_with_retry(
                            client,
                            f"{GOVINFO_API}/packages/{package_id}/granules/{g['granuleId']}/htm",
                            params={"api_key": api_key},
                        )
                    except httpx.HTTPStatusError as e:
                        print(f"[corpora] skip {g['granuleId']}: {e.response.status_code}")
                        continue
                    heading, body = _parse_usc_html(r.text)
                    if body:
                        records.append(
                            _usc_record(
                                g["granuleId"], title, year, heading or g.get("title", ""), body
                            )
                        )

    _write_jsonl(records, out)
    print(f"[corpora] wrote {len(records)} records -> {out}")
    return len(records)


# --- CourtListener ----------------------------------------------------------


def _opinion_text(data: dict[str, Any]) -> str:
    text = data.get("plain_text") or ""
    if not text:
        html = data.get("html_with_citations") or data.get("html") or ""
        text = re.sub(r"<[^>]+>", " ", html)
    return _ws(text)


def _cl_record(
    hit: dict[str, Any], op_meta: dict[str, Any], body: str, max_chars: int
) -> dict[str, Any] | None:
    text_body = body or _ws((hit.get("opinions") or [{}])[0].get("snippet", ""))
    if not text_body:
        return None
    if len(text_body) > max_chars:
        text_body = text_body[: max_chars - 1].rstrip() + "…"

    case_name = hit.get("caseName") or hit.get("caseNameFull") or ""
    cites = hit.get("citation") or []
    citation = (
        cites[0]
        if isinstance(cites, list) and cites
        else (cites if isinstance(cites, str) else case_name)
    ) or "?"
    op_id = (hit.get("opinions") or [{}])[0].get("id")
    cluster_id = hit.get("cluster_id", "")
    header = f"{case_name} ({hit.get('court_citation_string', '')}, {hit.get('dateFiled', '')})"
    abs_url = hit.get("absolute_url", "")
    return {
        "id": f"cl-op-{op_id or cluster_id}",
        "text": f"{header}\n\n{text_body}",
        "source": "courtlistener",
        "citation": str(citation),
        "type": "judgment",
        "metadata": {
            "case_name": case_name,
            "court": hit.get("court", ""),
            "court_id": hit.get("court_id", ""),
            "date_filed": hit.get("dateFiled", ""),
            "docket_number": hit.get("docketNumber", ""),
            "cluster_id": cluster_id,
            "opinion_id": op_id,
            "absolute_url": ("https://www.courtlistener.com" + abs_url) if abs_url else "",
            "fetched_full_body": bool(body),
            "author": op_meta.get("author_str", ""),
            "page_count": op_meta.get("page_count"),
        },
    }


def download_courtlistener(
    base_path: str | Path,
    query: str = "bank robbery OR fraud OR theft OR burglary OR assault OR murder OR forgery",
    court: str | None = None,
    max_docs: int = 50,
    api_token: str | None = None,
    page_size: int = 20,
    order_by: str = "score desc",
    full_bodies: bool = True,
    max_chars: int = 8000,
    pacing_seconds: float = 0.4,
) -> int:
    """Two-hop CL ingest: /search/?type=o ranks + metadata -> /opinions/{id}/ body.

    Falls back to snippet-only when no token is set. Token comes from arg or
    `COURTLISTENER_API_TOKEN` env (via .env). Honors 429 with backoff + Retry-After.
    """
    if api_token is None:
        load_env()
        api_token = os.environ.get("COURTLISTENER_API_TOKEN")
    out = Path(str(base_path) + ".jsonl")
    headers = dict(UA)
    if api_token:
        headers["Authorization"] = f"Token {api_token}"

    fetch_bodies = bool(full_bodies and api_token)
    if full_bodies and not api_token:
        print("[corpora] no COURTLISTENER_API_TOKEN -- snippets only.")

    params: dict | None = {"type": "o", "q": query, "page_size": page_size, "order_by": order_by}
    if court:
        params["court"] = court

    url = f"{COURTLISTENER_API}/search/"
    records: list[dict[str, Any]] = []
    with httpx.Client(headers=headers, timeout=60.0) as client:
        with tqdm(total=max_docs, desc="courtlistener") as bar:
            while url and len(records) < max_docs:
                payload = _get_with_retry(client, url, params=params).json()
                for hit in payload.get("results", []):
                    op_id = (hit.get("opinions") or [{}])[0].get("id")
                    body, op_meta = "", {}
                    if fetch_bodies and op_id:
                        try:
                            data = _get_with_retry(
                                client, f"{COURTLISTENER_API}/opinions/{op_id}/"
                            ).json()
                            body, op_meta = _opinion_text(data), data
                        except httpx.HTTPStatusError as e:
                            print(
                                f"[corpora] /opinions/{op_id} -> {e.response.status_code}; snippet"
                            )
                        if pacing_seconds > 0:
                            time.sleep(pacing_seconds)

                    rec = _cl_record(hit, op_meta, body, max_chars)
                    if rec is None:
                        continue
                    records.append(rec)
                    bar.update(1)
                    if len(records) >= max_docs:
                        break
                url = payload.get("next")
                params = None  # `next` carries its own querystring

    _write_jsonl(records, out)
    fb = sum(1 for r in records if r["metadata"]["fetched_full_body"])
    print(f"[corpora] wrote {len(records)} records ({fb} with full body) -> {out}")
    return len(records)


# --- UK legislation.gov.uk --------------------------------------------------


def _parse_uk_act(
    xml_bytes: bytes, act_type: str, year: int, number: int
) -> Iterator[dict[str, Any]]:
    """Yield one record per <P1> (section) from a whole-act CLML XML response."""
    from lxml import etree

    root = etree.fromstring(xml_bytes)
    ns = LEG_UK_NS
    title_el = root.find(".//dc:title", ns)
    act_title = _ws(title_el.text if title_el is not None else f"{act_type} {year} c.{number}")

    for p1 in root.iterfind(".//leg:Body//leg:P1", ns):
        num_el = p1.find("leg:Pnumber", ns)
        sec_num = _ws(num_el.text if num_el is not None else "")
        if not sec_num:
            continue
        # P1's heading sits in the parent <P1group>'s <Title> (when present).
        parent = p1.getparent()
        heading = ""
        if parent is not None and parent.tag.endswith("}P1group"):
            t = parent.find("leg:Title", ns)
            if t is not None:
                heading = _ws("".join(t.itertext()))
        body = _ws(" ".join(t for t in p1.itertext() if t))
        if not body:
            continue
        citation = f"{act_title}, s.{sec_num}"
        yield {
            "id": f"uk-{act_type}-{year}-{number}-s{sec_num}",
            "text": f"{citation}. {heading}\n\n{body}" if heading else f"{citation}\n\n{body}",
            "source": "uk_legislation",
            "citation": citation,
            "type": "statute",
            "metadata": {
                "act_title": act_title,
                "act_type": act_type,
                "year": year,
                "number": number,
                "section": sec_num,
                "heading": heading,
                "url": f"{LEG_UK}/{act_type}/{year}/{number}/section/{sec_num}",
            },
        }


def download_uk_legislation(
    base_path: str | Path,
    statutes: Iterable[tuple[str, int, int]] = UK_CRIMINAL_ACTS,
    max_sections_per_act: int | None = None,
) -> int:
    """Fetch UK Acts of Parliament from legislation.gov.uk and write JSONL.

    Each `statutes` entry is (act_type, year, number), e.g. ("ukpga", 2006, 35)
    = Fraud Act 2006. Pulls one whole-act XML per Act (CLML schema), iterates
    `<P1>` sections, and writes one record per section.

    Open Government Licence -- no key, no rate limit. ~1 HTTP request per Act.
    """
    statutes = list(statutes)
    records: list[dict[str, Any]] = []
    with httpx.Client(headers=UA, timeout=60.0) as client:
        for act_type, year, number in tqdm(statutes, desc="uk_legislation"):
            url = f"{LEG_UK}/{act_type}/{year}/{number}/data.xml"
            try:
                r = _get_with_retry(client, url)
            except httpx.HTTPStatusError as e:
                print(f"[corpora] skip {act_type}/{year}/{number}: {e.response.status_code}")
                continue
            act_records = list(_parse_uk_act(r.content, act_type, year, number))
            if max_sections_per_act is not None:
                act_records = act_records[:max_sections_per_act]
            records.extend(act_records)

    out = Path(str(base_path) + ".jsonl")
    _write_jsonl(records, out)
    print(f"[corpora] wrote {len(records)} records from {len(statutes)} Acts -> {out}")
    return len(records)


# --- BAILII (stub) ----------------------------------------------------------


def download_bailii(base_path: str | Path, **kwargs) -> int:
    raise NotImplementedError("BAILII ingestion not implemented yet. See https://www.bailii.org/ .")


# --- CLI --------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="crimellm.corpora", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pu = sub.add_parser("us-code", help="Ingest US Code via govinfo.gov.")
    pu.add_argument("--out", required=True)
    pu.add_argument("--titles", nargs="*", default=["18"])
    pu.add_argument("--year", type=int, default=2023)
    pu.add_argument("--api-key", default=None)
    pu.add_argument("--max-sections", type=int, default=None)
    pu.add_argument("--xml", default=None, help="Local USLM XML file/dir (skips API).")

    pc = sub.add_parser("courtlistener", help="Ingest CourtListener opinions.")
    pc.add_argument("--out", required=True)
    pc.add_argument(
        "--query",
        default="bank robbery OR fraud OR theft OR burglary OR assault OR murder OR forgery",
    )
    pc.add_argument("--court", default=None)
    pc.add_argument("--max-docs", type=int, default=50)
    pc.add_argument("--api-token", default=None)
    pc.add_argument("--order-by", default="score desc", choices=["score desc", "dateFiled desc"])
    pc.add_argument("--snippets-only", action="store_true")
    pc.add_argument("--max-chars", type=int, default=8000)

    pk = sub.add_parser("uk", help="Ingest UK Acts of Parliament from legislation.gov.uk.")
    pk.add_argument("--out", required=True)
    pk.add_argument(
        "--statutes",
        nargs="*",
        default=None,
        help='Slash-form ids like "ukpga/2006/35" (Fraud Act 2006). Default: UK_CRIMINAL_ACTS.',
    )
    pk.add_argument("--max-sections-per-act", type=int, default=None)

    a = p.parse_args(argv)
    if a.cmd == "us-code":
        download_us_code(
            a.out,
            titles=a.titles,
            year=a.year,
            api_key=a.api_key,
            max_sections=a.max_sections,
            xml_path=a.xml,
        )
    elif a.cmd == "courtlistener":
        download_courtlistener(
            a.out,
            query=a.query,
            court=a.court,
            max_docs=a.max_docs,
            api_token=a.api_token,
            order_by=a.order_by,
            full_bodies=not a.snippets_only,
            max_chars=a.max_chars,
        )
    elif a.cmd == "uk":
        statutes = UK_CRIMINAL_ACTS
        if a.statutes:
            statutes = tuple(
                (parts[0], int(parts[1]), int(parts[2]))
                for s in a.statutes
                for parts in [s.split("/")]
            )
        download_uk_legislation(
            a.out, statutes=statutes, max_sections_per_act=a.max_sections_per_act
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
