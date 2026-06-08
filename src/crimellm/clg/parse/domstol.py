"""domstol.dk judgment text → ``Case`` + citation hits.

DK court PDFs from the official portal vary in quality (selectable text
on modern judgments, scanned images on older ones). This parser splits
into two layers:

* ``parse_judgment_text`` — pure-text path. Takes already-extracted body
  text + operator-supplied metadata (ECLI, court id, decision date,
  case name). Picks up missing ECLI / decision-date from the header
  when the operator omits them. Runs the Phase 1 DK citation parser
  over the body and returns the hits as ``CitationHit`` rows. Zero deps
  beyond stdlib + ``crimellm.clg.link``.
* ``parse_judgment_pdf`` — PDF wrapper. Uses ``pypdf`` (added to the
  ``[clg]`` extra in Phase 5.5) to pull text; delegates everything else
  to ``parse_judgment_text``. OCR fallback for scanned PDFs is
  Phase 5.5+ work.

Court id mapping is hand-coded (4 entries — Højesteret, Østre Landsret,
Vestre Landsret, byret). The Court hierarchy gets seeded by the
``DK_COURTS`` constant in ``ingest/domstol.py`` so a fresh DB picks up
the nodes the moment the first DK judgment loads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..link import CitationHit, for_jurisdiction
from ..models import Case, Provenance

DOMSTOL_BASE = "https://domstol.dk"

# Recognised court ids — match the DK_COURTS seed in ingest/domstol.py.
COURT_IDS: tuple[str, ...] = ("hr", "olr", "vlr", "byret")

# Court id ← ECLI subjurisdiction code mapping. DK ECLIs use HR for
# Højesteret, OLR for Østre Landsret, VLR for Vestre Landsret.
_ECLI_COURT_TO_ID: dict[str, str] = {
    "HR": "hr",
    "OLR": "olr",
    "VLR": "vlr",
    "BYR": "byret",
}

# ECLI:DK pattern — court code captured for court-id lookup.
_ECLI_DK_RE = re.compile(r"\bECLI:DK:(?P<court>[A-Z]+):\d{4}:[A-Z0-9.]+\b")

# Danish date forms commonly seen in judgment headers:
#   "afsagt den 13. maj 2014" / "13. maj 2014" / "13/05-2014"
_DA_MONTHS: dict[str, int] = {
    "januar": 1,
    "februar": 2,
    "marts": 3,
    "april": 4,
    "maj": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
}
_DA_DATE_RE = re.compile(
    r"\b(?P<day>\d{1,2})\.?\s+(?P<month>januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|december)\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_SHORT_DATE_RE = re.compile(r"\b(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>\d{4})\b")
_ISO_DATE_RE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")


# --- low-level helpers ----------------------------------------------------


def _ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s).strip() if s else ""


def _extract_ecli(text: str) -> tuple[str, str] | None:
    """Return ``(ecli, court_id)`` of the first ECLI:DK match, else None."""
    m = _ECLI_DK_RE.search(text)
    if not m:
        return None
    court_code = m.group("court").upper()
    court_id = _ECLI_COURT_TO_ID.get(court_code, court_code.lower())
    return m.group(0), court_id


def _extract_decision_date(text: str) -> date | None:
    """Try Danish long-form first, then short DD/MM-YYYY, then ISO."""
    m = _DA_DATE_RE.search(text)
    if m:
        month = _DA_MONTHS.get(m.group("month").lower())
        if month is not None:
            try:
                return date(int(m.group("year")), month, int(m.group("day")))
            except ValueError:
                pass
    m = _SHORT_DATE_RE.search(text)
    if m:
        try:
            return date(int(m.group("year")), int(m.group("month")), int(m.group("day")))
        except ValueError:
            pass
    m = _ISO_DATE_RE.search(text)
    if m:
        try:
            return date(int(m.group("year")), int(m.group("month")), int(m.group("day")))
        except ValueError:
            pass
    return None


def court_id_for_ecli(ecli: str) -> str | None:
    """ECLI:DK court-code → Court node id slug.

    Returns ``None`` only when the input isn't an ECLI:DK at all.
    Unrecognised court codes (anything outside HR/OLR/VLR/BYR) get
    lowercased so a stray new code groups consistently rather than
    silently disappearing into a None.
    """
    m = _ECLI_DK_RE.match(ecli) or _ECLI_DK_RE.search(ecli)
    if not m:
        return None
    code = m.group("court").upper()
    return _ECLI_COURT_TO_ID.get(code, code.lower())


# --- main parse -----------------------------------------------------------


@dataclass(slots=True)
class JudgmentParse:
    """Result of parsing one DK judgment body."""

    case: Case
    body_text: str
    citation_hits: list[CitationHit] = field(default_factory=list)


def parse_judgment_text(
    body_text: str,
    *,
    ecli: str | None = None,
    court_id: str | None = None,
    decision_date: date | None = None,
    name: str | None = None,
    citations: list[str] | None = None,
    source_url: str | None = None,
    retrieved_at: date | None = None,
) -> JudgmentParse:
    """Parse a DK judgment from already-extracted text.

    Operator-supplied metadata wins over inference. When ``ecli`` is
    omitted we try to recover it from the first ``ECLI:DK:`` match in the
    body; ``court_id`` then derives from the ECLI court code. Same idea
    for ``decision_date`` against DA / short / ISO date patterns.
    """
    body_text = body_text or ""

    # Fill in missing identifiers from header text.
    if ecli is None or court_id is None:
        hit = _extract_ecli(body_text)
        if hit is not None:
            ecli = ecli or hit[0]
            court_id = court_id or hit[1]
    if ecli is None:
        raise ValueError(
            "ECLI not provided and not present in body text; "
            "pass ecli= explicitly or include 'ECLI:DK:...' in the body"
        )
    if court_id is None:
        court_id = "hr"  # safest default for unknown DK judgments

    if decision_date is None:
        decision_date = _extract_decision_date(body_text)

    if name is None:
        # First non-empty line is usually the case caption.
        for line in body_text.splitlines():
            line = line.strip()
            if line:
                name = line[:200]
                break
        name = name or ecli

    prov = Provenance(
        source="domstol.dk",
        source_url=source_url or DOMSTOL_BASE,
        retrieved_at=retrieved_at or date.today(),
        source_id=ecli,
    )

    case = Case(
        id=ecli,
        jurisdiction="DK",
        court_id=court_id,
        name=name,
        decision_date=decision_date,
        citations=list(citations or []),
        provenance=[prov],
    )

    # Lift DK + EU citation hits out of the body so the link phase can
    # MERGE CITES edges + IMPLEMENTS targets. Each registry parser
    # tolerates being called on text with no hits — empty list is fine.
    hits: list[CitationHit] = []
    for jur in ("DK", "EU"):
        parser = for_jurisdiction(jur)
        if parser is not None:
            hits.extend(parser.extract(body_text))

    return JudgmentParse(case=case, body_text=body_text, citation_hits=hits)


def parse_judgment_pdf(
    path: Path | str,
    *,
    ecli: str | None = None,
    court_id: str | None = None,
    decision_date: date | None = None,
    name: str | None = None,
    citations: list[str] | None = None,
    source_url: str | None = None,
    retrieved_at: date | None = None,
    allow_ocr: bool = True,
) -> JudgmentParse:
    """PDF wrapper. Extracts text via ``pypdf`` then delegates.

    Modern domstol.dk judgments carry a selectable text layer. Older
    ones are scans — ``pypdf`` returns empty pages. When that happens
    and ``allow_ocr=True`` (default) we try ``ocrmypdf`` as a fallback,
    rewriting the PDF in place with an embedded text layer and then
    re-extracting. The OCR escalation is no-op when the ``[ocr]`` extra
    isn't installed (``ocrmypdf`` import fails → we just return whatever
    ``pypdf`` could extract, however little).
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover — caller installs the extra
        raise ImportError(
            "pypdf not installed; add the [clg] extra "
            "(`uv sync --extra clg`) to parse domstol.dk PDFs."
        ) from e

    def _extract(pdf_path: Path | str) -> str:
        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 — encrypted / odd PDFs
                pages.append("")
        return "\n\n".join(pages)

    body = _extract(path)

    # Empty / near-empty extract on a non-trivial PDF → likely a scan.
    # Escalate to OCR when the optional extra is installed.
    if allow_ocr and len(body.strip()) < 40:
        ocr_body = _try_ocr_fallback(path)
        if ocr_body is not None:
            body = ocr_body

    return parse_judgment_text(
        body,
        ecli=ecli,
        court_id=court_id,
        decision_date=decision_date,
        name=name,
        citations=citations,
        source_url=source_url,
        retrieved_at=retrieved_at,
    )


def _try_ocr_fallback(path: Path | str) -> str | None:
    """Run ``ocrmypdf`` over ``path`` into a temp file + re-extract text.

    Returns ``None`` when the optional ``ocrmypdf`` extra isn't installed
    or the OCR pass fails — the caller falls back to whatever pypdf could
    pull (often empty). Skeleton: production use should pin Tesseract
    language data for DA (``ocrmypdf -l dan``) — wired below.
    """
    try:
        import ocrmypdf  # type: ignore
    except ImportError:
        return None

    import tempfile

    from pypdf import PdfReader

    src = Path(path)
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            out_path = Path(tmp.name)
        ocrmypdf.ocr(
            str(src),
            str(out_path),
            language="dan",
            deskew=True,
            skip_text=False,
            force_ocr=True,
            progress_bar=False,
        )
        reader = PdfReader(str(out_path))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                pages.append("")
        return "\n\n".join(pages)
    except Exception:  # noqa: BLE001 — OCR failures shouldn't crash ingest
        return None


def parse_judgment_file(path: Path | str, **kwargs: Any) -> JudgmentParse:
    """Auto-detect: ``.pdf`` → PDF path, anything else → text file."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return parse_judgment_pdf(p, **kwargs)
    return parse_judgment_text(p.read_text(encoding="utf-8"), **kwargs)
