"""Retsinformation XML → ``Instrument`` + ``Provision``.

Retsinformation publishes Danish primary law (love, lovbekendtgørelser,
bekendtgørelser) via the harvest API at ``api.retsinformation.dk`` with
per-document XML at:

    http://retsinformation.dk/eli/accn/<ACCN>/xml

The XML serialisation is the **LexDania** schema (Civilstyrelsen's XML
DTD family). For Phase 4 we parse the subset that's stable across
document types: ``<Dokument>`` root with ``<Meta>`` header +
``<DokumentIndhold>`` body containing ``<Paragraf localId="N">`` (=§)
→ ``<Stk>`` (=stykke) → ``<Indentatio formaInd="Nummer">`` (=nr)
elements. Text bodies live in ``<Exitus><Linea><Char>...</Char></Linea>``.

For backward compatibility this module also handles an older synthetic
schema (lower-case ``<dokument>/<paragraf>/<stk>/<nr>``) used in some
tests / hand-saved exports. The two paths are routed by root tag.

Identifiers:

* ``Instrument.id`` = ``dk/<doc_type>/<year>/<num>`` (e.g. ``dk/lbk/2018/502``
  for Databeskyttelsesloven). Stable across consolidations (``lbk``
  republications get a new (year, num); the Instrument id changes with
  them because consolidated republication semantically *is* a new
  Instrument under DK practice).
* ``Provision.id`` = ``dk/<doc_type>/<year>/<num>/section/<§N>[/stk.M][/nr.K]``
  with the same path shape as the DK citation parser emits (Phase 1.3) so
  cross-references collapse cleanly under the same node.
* ``Provision.section_path`` = ``§ N`` / ``§ N stk. M`` / ``§ N stk. M nr. K``,
  human-readable.

Preamble extraction returns CELEX ids derived from DK references to EU
directives / regulations (``direktiv 2019/770`` → ``32019L0770``;
``forordning (EU) 2016/679`` → ``32016R0679``). These feed
``load_implements`` so a DK lbk implementing an EU directive gets a
``(Instrument)-[:IMPLEMENTS]->(Instrument)`` edge into the EU subgraph
ingested in Phase 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from lxml import etree

from ..models import Instrument, Provenance, Provision

RETSINFO_BASE = "https://www.retsinformation.dk"

# Document types in the Retsinformation taxonomy. ``lov`` = act, ``lbk`` =
# consolidated act (lovbekendtgørelse), ``bek`` = executive order, ``ltc`` =
# constitutional, ``vejledning`` = guidance. Phase 4 focuses on the first
# three — they're the primary-law surface DA lawyers actually cite.
DOC_TYPES: tuple[str, ...] = ("lov", "lbk", "bek", "ltc", "vejledning")

# Retsinformation custom namespace. Operators may serve XML with or without
# the namespace declared — we use local-name XPath everywhere to tolerate
# both forms (mirrors the AKN parser in parse/eurlex.py).
RETSINFO_NS = "https://www.retsinformation.dk/ns/dokument"


# --- EU-directive ↔ CELEX helpers -----------------------------------------

# DK preambles cite EU acts in many surface forms. The common ones we
# convert to CELEX:
#   "direktiv 2019/770"                         → 32019L0770
#   "Europa-Parlamentets og Rådets direktiv 95/46/EF" → 31995L0046
#   "forordning (EU) 2016/679"                  → 32016R0679
#   "forordning (EF) nr. 45/2001"               → 32001R0045
#   "Rådets forordning (EU) 2016/679"           → 32016R0679
_DK_DIRECTIVE_RE = re.compile(
    r"\bdirektiv\s+(?:\(?(?:EU|EF|EØF)\)?\s+(?:nr\.?\s+)?)?(?P<year>\d{2,4})/(?P<num>\d{1,5})(?:/(?:EU|EF|EØF))?\b",
    re.IGNORECASE,
)
_DK_REGULATION_RE = re.compile(
    r"\bforordning\s+\(?(?:EU|EF|EØF)?\)?\s+(?:nr\.?\s+)?(?P<year>\d{2,4})/(?P<num>\d{1,5})\b",
    re.IGNORECASE,
)
# Already-canonical CELEX (e.g. when the preamble lists footnote ids).
_RAW_CELEX_RE = re.compile(r"\b[1-9]\d{4}[A-Z]{1,2}\d{4}\b")


def _normalise_year(year: str) -> str:
    """Two-digit DK year refs map to 19XX (pre-2000 directives)."""
    if len(year) == 2:
        return f"19{year}"
    return year


def directive_to_celex(year: str, num: str) -> str:
    """``("2019","770")`` → ``"32019L0770"``."""
    return f"3{_normalise_year(year)}L{int(num):04d}"


def regulation_to_celex(year: str, num: str) -> str:
    """``("2016","679")`` → ``"32016R0679"``."""
    return f"3{_normalise_year(year)}R{int(num):04d}"


def extract_eu_celex_refs(text: str) -> list[str]:
    """Pull DK-style EU references out of free text, return CELEX list.

    Document-order, deduped. Includes already-canonical CELEX surface forms
    so an operator pasting "32019L0770" into a preamble gets picked up.
    """
    seen: set[str] = set()
    hits: list[tuple[int, str]] = []
    for m in _DK_DIRECTIVE_RE.finditer(text):
        c = directive_to_celex(m["year"], m["num"])
        if c not in seen:
            seen.add(c)
            hits.append((m.start(), c))
    for m in _DK_REGULATION_RE.finditer(text):
        c = regulation_to_celex(m["year"], m["num"])
        if c not in seen:
            seen.add(c)
            hits.append((m.start(), c))
    for m in _RAW_CELEX_RE.finditer(text):
        c = m.group(0)
        if c not in seen:
            seen.add(c)
            hits.append((m.start(), c))
    hits.sort(key=lambda t: t[0])
    return [c for _, c in hits]


# --- identifier helpers ---------------------------------------------------


def instrument_id(doc_type: str, year: int, num: int) -> str:
    return f"dk/{doc_type.lower()}/{year}/{num}"


def section_path(par_num: str, stk: str | None = None, nr: str | None = None) -> str:
    """Human-readable Danish section path: ``§ 36 stk. 2 nr. 1``."""
    parts = [f"§ {par_num.strip()}"]
    if stk:
        parts.append(f"stk. {stk.strip()}")
    if nr:
        parts.append(f"nr. {nr.strip()}")
    return " ".join(parts)


def provision_id(
    doc_type: str,
    year: int,
    num: int,
    par_num: str,
    stk: str | None = None,
    nr: str | None = None,
) -> str:
    """Slash-path mirror of the DK citation parser's normalised id."""
    parts = [f"{instrument_id(doc_type, year, num)}/section/§{par_num.strip().lower()}"]
    if stk:
        parts.append(f"stk.{stk.strip()}")
    if nr:
        parts.append(f"nr.{nr.strip()}")
    return "/".join(parts)


def eli_url(doc_type: str, year: int, num: int) -> str:
    return f"{RETSINFO_BASE}/eli/{doc_type.lower()}/{year}/{num}"


# --- low-level helpers ----------------------------------------------------


def _ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s).strip() if s else ""


def _date_or_none(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _itertext(el: etree._Element) -> str:
    return _ws(" ".join(t for t in el.itertext() if t))


def _findall_local(root: etree._Element, local: str) -> list[etree._Element]:
    return root.xpath(f".//*[local-name()='{local}']")


def _find_local(root: etree._Element, local: str) -> etree._Element | None:
    hits = root.xpath(f".//*[local-name()='{local}']")
    return hits[0] if hits else None


def _meta_text(root: etree._Element, name: str) -> str:
    """Read ``<meta name="..."/>`` style headers if present."""
    for el in _findall_local(root, "meta"):
        if (el.get("name") or "").lower() == name.lower():
            return el.get("value") or ""
    el = _find_local(root, name)
    if el is not None:
        return _itertext(el)
    return ""


# --- main parse -----------------------------------------------------------


@dataclass(slots=True)
class StatuteParse:
    """Result of parsing one Retsinformation XML body."""

    instrument: Instrument
    provisions: list[Provision] = field(default_factory=list)
    # CELEX ids the preamble cites — drives IMPLEMENTS-edge seeding from
    # DK lbk → EU directive/regulation.
    cites_eu_celex: list[str] = field(default_factory=list)


def parse_statute(
    xml_bytes: bytes,
    *,
    doc_type: str,
    year: int,
    num: int,
    source_url: str | None = None,
    retrieved_at: date | None = None,
    explode_subparagraphs: bool = True,
) -> StatuteParse:
    """Parse one Retsinformation statute XML body.

    ``doc_type / year / num`` form the canonical Instrument id; pass them
    explicitly because the surface form in the XML header varies across
    Civilstyrelsen template versions.

    ``explode_subparagraphs`` controls whether stk./nr. become separate
    ``Provision`` nodes (True, default) or get folded into the parent §
    text (False). True gives finer-grained chunks for retrieval; False is
    cheaper at index time.

    Auto-routes between the real **LexDania** schema served by the live
    Retsinformation API (``<Dokument>`` root) and an older synthetic
    schema (lowercase ``<dokument>`` root) kept around for back-compat
    fixtures.
    """
    if doc_type.lower() not in DOC_TYPES:
        raise ValueError(f"unknown DK doc_type {doc_type!r}; pick from {DOC_TYPES}")

    root = etree.fromstring(xml_bytes)

    # Dispatch on root tag name (LexDania uses CapWord, synthetic uses lowercase).
    if root.tag == "Dokument":
        return _parse_lexdania(
            root,
            doc_type=doc_type,
            year=year,
            num=num,
            source_url=source_url,
            retrieved_at=retrieved_at,
            explode_subparagraphs=explode_subparagraphs,
        )

    title = _meta_text(root, "titel") or _meta_text(root, "title")
    # Try common surface forms for publication / consolidation date.
    publish = (
        _date_or_none(_meta_text(root, "publikationsdato"))
        or _date_or_none(_meta_text(root, "ikrafttraedelse"))
        or _date_or_none(_meta_text(root, "publication_date"))
    )

    iid = instrument_id(doc_type, year, num)
    prov = Provenance(
        source="retsinformation",
        source_url=source_url or eli_url(doc_type, year, num),
        retrieved_at=retrieved_at or date.today(),
        source_id=iid,
    )

    instrument = Instrument(
        id=iid,
        jurisdiction="DK",
        short_title=title or iid,
        year=year,
        provenance=[prov],
    )

    provisions: list[Provision] = []
    for paragraf in _findall_local(root, "paragraf"):
        par_num = (
            paragraf.get("nr")
            or _itertext(next(iter(paragraf.xpath("./*[local-name()='nr']")), None))
            if next(iter(paragraf.xpath("./*[local-name()='nr']")), None) is not None
            else paragraf.get("nr")
        )
        # Fallback when number lives in a child element.
        if not par_num:
            num_el = next(iter(paragraf.xpath("./*[local-name()='paragraf-nr']")), None)
            par_num = _itertext(num_el) if num_el is not None else ""
        if not par_num:
            continue
        par_num = par_num.strip().lstrip("§").strip()

        heading_el = next(iter(paragraf.xpath("./*[local-name()='overskrift']")), None)
        heading = _itertext(heading_el) if heading_el is not None else ""

        stk_elements = paragraf.xpath("./*[local-name()='stk']")
        if not stk_elements or not explode_subparagraphs:
            # Flat §: one Provision for the whole paragraph.
            text = _itertext(paragraf)
            if heading:
                text = f"§ {par_num} — {heading}\n\n{text}".strip()
            provisions.append(
                Provision(
                    id=provision_id(doc_type, year, num, par_num),
                    instrument_id=iid,
                    jurisdiction="DK",
                    section_path=section_path(par_num),
                    text=text,
                    valid_from=publish,
                    valid_to=None,
                    version_id=None,
                )
            )
            continue

        for stk_el in stk_elements:
            stk_num = stk_el.get("nr") or ""
            nr_elements = stk_el.xpath("./*[local-name()='nr']")
            if nr_elements:
                for nr_el in nr_elements:
                    nr_num = nr_el.get("nr") or ""
                    text = _itertext(nr_el)
                    head = section_path(par_num, stk_num or None, nr_num or None)
                    body = f"{head}\n\n{text}" if text else head
                    provisions.append(
                        Provision(
                            id=provision_id(
                                doc_type, year, num, par_num, stk_num or None, nr_num or None
                            ),
                            instrument_id=iid,
                            jurisdiction="DK",
                            section_path=head,
                            text=body,
                            valid_from=publish,
                            valid_to=None,
                            version_id=None,
                        )
                    )
            else:
                text = _itertext(stk_el)
                head = section_path(par_num, stk_num or None)
                body = f"{head}\n\n{text}" if text else head
                provisions.append(
                    Provision(
                        id=provision_id(doc_type, year, num, par_num, stk_num or None),
                        instrument_id=iid,
                        jurisdiction="DK",
                        section_path=head,
                        text=body,
                        valid_from=publish,
                        valid_to=None,
                        version_id=None,
                    )
                )

    # Preamble — DK XMLs usually wrap recitals in <praeambel> / <indledning>
    # or scatter notes near the top. Fall back to scanning header text if
    # neither is present.
    preamble_blobs: list[str] = []
    for tag in ("praeambel", "indledning", "noter"):
        el = _find_local(root, tag)
        if el is not None:
            preamble_blobs.append(_itertext(el))
    if not preamble_blobs:
        preamble_blobs.append(_itertext(root))
    cites: list[str] = []
    seen: set[str] = set()
    for blob in preamble_blobs:
        for c in extract_eu_celex_refs(blob):
            if c not in seen:
                seen.add(c)
                cites.append(c)

    return StatuteParse(instrument=instrument, provisions=provisions, cites_eu_celex=cites)


def parse_statute_file(path: Path | str, **kwargs: Any) -> StatuteParse:
    raw = Path(path).read_bytes()
    return parse_statute(raw, **kwargs)


# --- LexDania (real Retsinformation API schema) --------------------------


_STK_NUM_RE = re.compile(r"Stk\.\s*(\d+)", re.IGNORECASE)
_NR_NUM_RE = re.compile(r"^(\d+[a-z]?)\)")


def _explicatus_text(parent: etree._Element) -> str:
    """Return the direct ``<Explicatus>`` child's text (e.g. ``§ 1.``, ``Stk. 2.``, ``1)``)."""
    for child in parent:
        if child.tag == "Explicatus":
            return _itertext(child)
    return ""


def _exitus_text(parent: etree._Element) -> str:
    """Concatenate all ``<Exitus>...</Exitus>`` direct/child text inside ``parent``,
    skipping any nested ``<Indentatio>`` blocks (those become separate Provisions)."""
    parts: list[str] = []
    for exitus in parent.iterfind("Exitus"):
        # Drop nested Indentatio so the parent body doesn't double-count.
        for sub in list(exitus):
            if sub.tag == "Index":
                exitus.remove(sub)
        t = _itertext(exitus)
        if t:
            parts.append(t)
    return _ws(" ".join(parts))


def _stk_number(stk_el: etree._Element, fallback_idx: int) -> str:
    """Extract a stk. number — from ``<Explicatus>`` or by document-order index."""
    expl = _explicatus_text(stk_el)
    m = _STK_NUM_RE.search(expl)
    if m:
        return m.group(1)
    return str(fallback_idx + 1)


def _nr_number(nr_el: etree._Element, fallback_idx: int) -> str:
    """Extract a nr. number from ``<Indentatio><Explicatus>1)</Explicatus>``."""
    expl = _explicatus_text(nr_el)
    m = _NR_NUM_RE.match(expl.strip())
    if m:
        return m.group(1)
    return str(fallback_idx + 1)


def _parse_lexdania(
    root: etree._Element,
    *,
    doc_type: str,
    year: int,
    num: int,
    source_url: str | None,
    retrieved_at: date | None,
    explode_subparagraphs: bool,
) -> StatuteParse:
    """Real LexDania parser. ``<Dokument><Meta>`` + ``<DokumentIndhold>`` body."""
    meta = root.find("Meta")

    def _meta(tag: str) -> str:
        if meta is None:
            return ""
        el = meta.find(tag)
        return _itertext(el) if el is not None else ""

    accn = _meta("AccessionNumber")
    title = _meta("PopularTitle") or _meta("DocumentTitle")
    publish = (
        _date_or_none(_meta("DiesEdicti"))
        or _date_or_none(_meta("StartDate"))
        or _date_or_none(_meta("DiesSigni"))
    )
    end_date = _date_or_none(_meta("EndDate"))
    # End-of-time sentinel in LexDania (9999-12-31) → None for valid_to.
    if end_date and end_date.year >= 9999:
        end_date = None

    iid = instrument_id(doc_type, year, num)
    prov = Provenance(
        source="retsinformation",
        source_url=source_url or (
            f"http://retsinformation.dk/eli/accn/{accn}/xml" if accn else eli_url(doc_type, year, num)
        ),
        retrieved_at=retrieved_at or date.today(),
        source_id=accn or iid,
    )

    instrument = Instrument(
        id=iid,
        jurisdiction="DK",
        short_title=title or iid,
        year=year,
        provenance=[prov],
    )

    provisions: list[Provision] = []
    indhold = root.find("DokumentIndhold")
    if indhold is not None:
        for paragraf in indhold.iter("Paragraf"):
            par_num = (paragraf.get("localId") or "").strip()
            if not par_num:
                # Try Explicatus ("§ 1.") fallback.
                expl = _explicatus_text(paragraf)
                m = re.search(r"§\s*(\d+[a-z]?)", expl)
                par_num = m.group(1) if m else ""
            if not par_num:
                continue

            stk_elements = list(paragraf.iterfind("Stk"))
            if not stk_elements or not explode_subparagraphs:
                # Flat §: one Provision for the whole paragraph.
                text = _itertext(paragraf)
                provisions.append(
                    Provision(
                        id=provision_id(doc_type, year, num, par_num),
                        instrument_id=iid,
                        jurisdiction="DK",
                        section_path=section_path(par_num),
                        text=text,
                        valid_from=publish,
                        valid_to=end_date,
                        version_id=None,
                    )
                )
                continue

            for stk_idx, stk_el in enumerate(stk_elements):
                stk_num = _stk_number(stk_el, stk_idx)
                nr_elements = list(stk_el.iterfind(".//Indentatio[@formaInd='Nummer']"))
                if nr_elements:
                    for nr_idx, nr_el in enumerate(nr_elements):
                        nr_num = _nr_number(nr_el, nr_idx)
                        text = _itertext(nr_el)
                        head = section_path(par_num, stk_num, nr_num)
                        body = f"{head}\n\n{text}" if text else head
                        provisions.append(
                            Provision(
                                id=provision_id(doc_type, year, num, par_num, stk_num, nr_num),
                                instrument_id=iid,
                                jurisdiction="DK",
                                section_path=head,
                                text=body,
                                valid_from=publish,
                                valid_to=end_date,
                                version_id=None,
                            )
                        )
                else:
                    text = _exitus_text(stk_el) or _itertext(stk_el)
                    head = section_path(par_num, stk_num)
                    body = f"{head}\n\n{text}" if text else head
                    provisions.append(
                        Provision(
                            id=provision_id(doc_type, year, num, par_num, stk_num),
                            instrument_id=iid,
                            jurisdiction="DK",
                            section_path=head,
                            text=body,
                            valid_from=publish,
                            valid_to=end_date,
                            version_id=None,
                        )
                    )

    # LexDania <Meta><EuReferences>CELEX</EuReferences> gives us the IMPLEMENTS
    # seeds **directly**, no preamble regex needed. Also scan the body as a
    # fallback for documents that embed CELEX inline.
    cites: list[str] = []
    seen: set[str] = set()
    if meta is not None:
        for ref in meta.iterfind("EuReferences"):
            celex = _ws(ref.text or "")
            if celex and celex not in seen:
                seen.add(celex)
                cites.append(celex)
    if indhold is not None:
        for c in extract_eu_celex_refs(_itertext(indhold)):
            if c not in seen:
                seen.add(c)
                cites.append(c)

    return StatuteParse(instrument=instrument, provisions=provisions, cites_eu_celex=cites)
