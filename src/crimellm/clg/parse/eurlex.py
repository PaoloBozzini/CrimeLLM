"""EUR-Lex / CELLAR XML → ``Instrument`` + ``Provision`` + ``Case``.

Targets the **Akoma Ntoso 3.0** schema EU publications are migrating to
(``http://docs.oasis-open.org/legaldocml/ns/akn/3.0``). FORMEX legacy XML
is a follow-up — same logical shape (FRBR work/expression, body of
articles), different element names.

For each ingested document we resolve:

* **CELEX** — the universal EU id, e.g. ``32016R0679`` (GDPR). Becomes part
  of the canonical node id (``eu/celex/<CELEX>``).
* **ELI** — slash-path URI when present (``/eli/reg/2016/679``); stored as
  ``Instrument.id`` alt when CELEX is missing.
* **ECLI** — case law (``ECLI:EU:C:2016:316``). Becomes ``Case.id``.
* **Adoption / judgment date** — from ``<FRBRdate>`` in the FRBR block.

Multilingual policy (Phase 3): one language ingested per file → last write
wins on duplicate CELEX. Operator runs the ingester per language; EN is
the cross-border default, DA is layered on top when needed. Each
language's source URL lands in ``Provenance.source_url`` so the audit
trail shows which body was indexed. Full per-language Provision splits
are deferred to Phase 3.5.

Preamble extraction: ``parse_regulation`` also returns the set of CELEX
ids the regulation cites in its recitals / footnotes — used to seed the
``(Instrument)-[:IMPLEMENTS]->(Instrument)`` edge from DK transposing
legislation to its EU source act (Phase 4 wires the reverse direction
when Retsinformation lbk's preamble names the EU directive).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from lxml import etree

from ..models import Case, Instrument, Provenance, Provision

EURLEX_BASE = "https://eur-lex.europa.eu"
CELLAR_BASE = "https://publications.europa.eu/resource/celex"

# Akoma Ntoso 3.0 namespace. Older EU bodies sometimes use the unversioned
# ``http://docs.oasis-open.org/legaldocml/ns/akn`` form — we accept both
# via a wildcard local-name match in XPath helpers.
AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
NS = {"akn": AKN_NS}

# CELEX matches anywhere in preamble / footnotes — used for IMPLEMENTS edge.
_CELEX_RE = re.compile(r"\b[1-9]\d{4}[A-Z]{1,2}\d{4}\b")
_ECLI_RE = re.compile(r"\bECLI:EU:[CTF]:\d{4}:\d+\b")


# --- identifier helpers ---------------------------------------------------


def instrument_id_from_celex(celex: str) -> str:
    """``32016R0679`` → ``eu/celex/32016R0679``. Stable canonical id."""
    return f"eu/celex/{celex.strip()}"


def provision_id(celex: str, article: str) -> str:
    """One Provision per article. Stays language-agnostic."""
    return f"{instrument_id_from_celex(celex)}/article/{article.strip()}"


def case_id_from_ecli(ecli: str) -> str:
    """ECLI is already canonical — use it verbatim as ``Case.id``."""
    return ecli.strip()


def celex_kind(celex: str) -> str:
    """First sector digit → document kind.

    ``3`` = legislation (regulations, directives, decisions). ``6`` = case
    law. Others (``1`` treaties, ``2`` international agreements, etc.) fall
    through as ``"other"``.
    """
    if not celex:
        return "other"
    first = celex[0]
    if first == "3":
        return "legislation"
    if first == "6":
        return "case"
    return "other"


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
    """XPath by local-name so we tolerate the unversioned AKN namespace."""
    return root.xpath(f".//*[local-name()='{local}']")


def _find_local(root: etree._Element, local: str) -> etree._Element | None:
    hits = root.xpath(f".//*[local-name()='{local}']")
    return hits[0] if hits else None


def _frbr_alias(root: etree._Element, name: str) -> str | None:
    """``<FRBRalias name="CELEX" value="..."/>`` → ``value``."""
    for el in _findall_local(root, "FRBRalias"):
        if (el.get("name") or "").lower() == name.lower():
            return el.get("value")
    return None


def _frbr_date(root: etree._Element, name: str) -> date | None:
    """``<FRBRdate date="2016-04-27" name="adoption"/>`` → ``date``."""
    for el in _findall_local(root, "FRBRdate"):
        if (el.get("name") or "").lower() == name.lower():
            return _date_or_none(el.get("date"))
    # Some bodies omit the name attribute — first FRBRdate wins as fallback.
    for el in _findall_local(root, "FRBRdate"):
        d = _date_or_none(el.get("date"))
        if d is not None:
            return d
    return None


def _doc_title(root: etree._Element) -> str:
    for local in ("docTitle", "docShortTitle", "docNumber"):
        el = _find_local(root, local)
        if el is not None:
            t = _itertext(el)
            if t:
                return t
    return ""


# --- regulation / directive parser ----------------------------------------


@dataclass(slots=True)
class RegulationParse:
    """Result of parsing one Akoma Ntoso legislation file."""

    instrument: Instrument
    provisions: list[Provision] = field(default_factory=list)
    # CELEX ids the preamble / recitals cite. Drives IMPLEMENTS-edge seeding
    # — DK lbk preamble referring to ``32019L0770`` etc.
    cites_celex: list[str] = field(default_factory=list)


def parse_regulation(
    xml_bytes: bytes,
    *,
    celex: str | None = None,
    language: str = "en",
    source_url: str | None = None,
    retrieved_at: date | None = None,
) -> RegulationParse:
    """Parse one Akoma Ntoso regulation / directive XML body.

    ``celex`` is the canonical EU id (e.g. ``32016R0679``). When omitted we
    try to recover it from ``<FRBRalias name="CELEX"/>``. ``language`` is
    the ISO 639-1 code of the body's content language; stored on
    ``Provenance`` for audit.
    """
    root = etree.fromstring(xml_bytes)

    celex = celex or _frbr_alias(root, "CELEX") or ""
    if not celex:
        raise ValueError("CELEX not provided and not present in <FRBRalias>")

    adoption = (
        _frbr_date(root, "adoption")
        or _frbr_date(root, "publication")
        or _frbr_date(root, "")
    )
    year = adoption.year if adoption else None
    title = _doc_title(root)

    iid = instrument_id_from_celex(celex)
    prov = Provenance(
        source="eur-lex",
        source_url=source_url or f"{CELLAR_BASE}/{celex}",
        retrieved_at=retrieved_at or date.today(),
        source_id=f"{celex}@{language}",
    )

    instrument = Instrument(
        id=iid,
        jurisdiction="EU",
        short_title=title or celex,
        year=year,
        provenance=[prov],
    )

    provisions: list[Provision] = []
    # Articles can sit anywhere under <body>; use local-name so we don't
    # care about AKN namespace version.
    body = _find_local(root, "body")
    if body is not None:
        for art in body.xpath(".//*[local-name()='article']"):
            num_el = next(iter(art.xpath("./*[local-name()='num']")), None)
            heading_el = next(iter(art.xpath("./*[local-name()='heading']")), None)
            num = _itertext(num_el) if num_el is not None else ""
            heading = _itertext(heading_el) if heading_el is not None else ""
            if not num:
                # Skip articles missing a number — usually transitional
                # placeholders that the body still renders.
                continue
            article_path = _article_path(num)
            body_text = _article_body_text(art)
            display_head = (
                f"{article_path} — {heading}" if heading else article_path
            )
            text = f"{display_head}\n\n{body_text}".strip() if body_text else display_head
            provisions.append(
                Provision(
                    id=provision_id(celex, article_path),
                    instrument_id=iid,
                    jurisdiction="EU",
                    section_path=article_path,
                    text=text,
                    valid_from=adoption,
                    valid_to=None,
                    version_id=language,
                )
            )

    # CELEX references inside preamble / recitals → IMPLEMENTS-seed list.
    cites: list[str] = []
    seen: set[str] = set()
    preamble = _find_local(root, "preamble")
    haystacks: list[str] = []
    if preamble is not None:
        haystacks.append(_itertext(preamble))
    for fn in _findall_local(root, "authorialNote"):
        haystacks.append(_itertext(fn))
    for blob in haystacks:
        for hit in _CELEX_RE.findall(blob):
            if hit != celex and hit not in seen:
                seen.add(hit)
                cites.append(hit)

    return RegulationParse(instrument=instrument, provisions=provisions, cites_celex=cites)


def _article_path(num_text: str) -> str:
    """``Article 6`` / ``Article 6(1)`` → ``art.6`` / ``art.6(1)``."""
    cleaned = re.sub(r"^(article|art\.?)\s*", "", num_text, flags=re.IGNORECASE).strip()
    if not cleaned:
        cleaned = num_text.strip()
    return f"art.{cleaned}"


def _article_body_text(article_el: etree._Element) -> str:
    """Concatenate paragraph / subparagraph text in document order."""
    # Skip the article's <num>/<heading> children — they're the display head.
    body_parts: list[str] = []
    for child in article_el:
        local = etree.QName(child).localname
        if local in {"num", "heading"}:
            continue
        body_parts.append(_itertext(child))
    return _ws(" ".join(p for p in body_parts if p))


# --- CJEU judgment parser -------------------------------------------------


@dataclass(slots=True)
class JudgmentParse:
    """Result of parsing one Akoma Ntoso CJEU judgment file."""

    case: Case
    body_text: str
    cites_ecli: list[str] = field(default_factory=list)
    cites_celex: list[str] = field(default_factory=list)


def parse_judgment(
    xml_bytes: bytes,
    *,
    ecli: str | None = None,
    celex: str | None = None,
    court_id: str = "cjeu",
    language: str = "en",
    source_url: str | None = None,
    retrieved_at: date | None = None,
) -> JudgmentParse:
    """Parse one Akoma Ntoso judgment XML body.

    ``ecli`` defaults to the ``<FRBRalias name="ECLI"/>`` value. ``celex``
    likewise from the alias block. ``court_id`` is the slug for the
    Court node (``cjeu`` for the Court of Justice; ``gc`` for the General
    Court when we add it).
    """
    root = etree.fromstring(xml_bytes)

    ecli = ecli or _frbr_alias(root, "ECLI") or ""
    celex = celex or _frbr_alias(root, "CELEX") or ""
    if not ecli:
        raise ValueError("ECLI not provided and not present in <FRBRalias>")

    decision_date = (
        _frbr_date(root, "judgment")
        or _frbr_date(root, "publication")
        or _frbr_date(root, "")
    )
    title = _doc_title(root)

    prov = Provenance(
        source="eur-lex",
        source_url=source_url
        or (f"{CELLAR_BASE}/{celex}" if celex else f"{EURLEX_BASE}/legal-content/{ecli}"),
        retrieved_at=retrieved_at or date.today(),
        source_id=f"{ecli}@{language}",
    )

    citations_alt = [celex] if celex else []
    case = Case(
        id=case_id_from_ecli(ecli),
        jurisdiction="EU",
        court_id=court_id,
        name=title or ecli,
        decision_date=decision_date,
        citations=citations_alt,
        provenance=[prov],
    )

    body = _find_local(root, "body")
    body_text = _itertext(body) if body is not None else ""

    haystack = body_text
    ecli_hits: list[str] = []
    seen_e: set[str] = set()
    for hit in _ECLI_RE.findall(haystack):
        if hit != ecli and hit not in seen_e:
            seen_e.add(hit)
            ecli_hits.append(hit)
    celex_hits: list[str] = []
    seen_c: set[str] = set()
    for hit in _CELEX_RE.findall(haystack):
        if hit != celex and hit not in seen_c:
            seen_c.add(hit)
            celex_hits.append(hit)

    return JudgmentParse(
        case=case,
        body_text=body_text,
        cites_ecli=ecli_hits,
        cites_celex=celex_hits,
    )


# --- file-path conveniences -----------------------------------------------


def parse_regulation_file(path: Path | str, **kwargs: Any) -> RegulationParse:
    raw = Path(path).read_bytes()
    return parse_regulation(raw, **kwargs)


def parse_judgment_file(path: Path | str, **kwargs: Any) -> JudgmentParse:
    raw = Path(path).read_bytes()
    return parse_judgment(raw, **kwargs)
