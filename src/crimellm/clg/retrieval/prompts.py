"""Language-routed synthesis prompts + citation format helpers.

Two surfaces, one module:

1. **Prompts** — ``system_prompt_for(language)`` and ``disclaimer_for(language)``
   return the right text for the user's query language. EN keeps the
   common-law framing (binding precedent, distinguishing cases, etc.); DA
   uses civil-law framing (Højesteret praksis is persuasive weight, not
   binding precedent — DA judges fravige rather than overrule).
2. **Citation formatting** — ``format_human_citation(candidate, language)``
   produces a human-readable heading used in the prompt context block.
   The bracketed ``[<parent_id>]`` identifier guard is **untouched** —
   every claim must still cite the canonical id verbatim; the formatter
   only enriches the surrounding prose so a reader sees "GDPR Art. 6
   [eu/celex/32016R0679/article/art.6]" instead of the raw path twice.

Add a language by writing two strings (system prompt + disclaimer) and
extending the format helpers if the jurisdiction's id scheme needs
special prose. The ``Synthesizer`` ABC and citation guard don't care
about language.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-only import to avoid a cycle
    from .seed import Candidate


# --- disclaimers ----------------------------------------------------------


DISCLAIMER_EN = (
    "This is research support, not legal advice. Every statement is grounded "
    "in the retrieved authorities; consult a qualified lawyer for advice on "
    "any specific matter."
)

DISCLAIMER_DA = (
    "Dette er forskningsstøtte, ikke juridisk rådgivning. Alle udsagn er "
    "forankret i de fundne kilder; kontakt en kvalificeret jurist for "
    "rådgivning i en konkret sag. Verificer alle citater før brug."
)

_DISCLAIMERS: dict[str, str] = {
    "en": DISCLAIMER_EN,
    "da": DISCLAIMER_DA,
}


def disclaimer_for(language: str | None) -> str:
    """Return the standing disclaimer for ``language`` (EN fallback)."""
    if not language:
        return DISCLAIMER_EN
    return _DISCLAIMERS.get(language.lower(), DISCLAIMER_EN)


# --- system prompts -------------------------------------------------------


# Common-law framing: precedent, distinguishing, overruling, "still good law".
SYSTEM_PROMPT_EN = """You are a research assistant for primary-law questions.

Strict rules:
1. Use ONLY the passages provided under "Context" below as evidence.
2. Cite every factual claim by the bracketed identifier of its source —
   exactly as it appears in the context, e.g. ``[uk/ukpga/2006/35/section/2@enacted]``,
   ``[eu/celex/32016R0679/article/art.6]``, or ``[ECLI:DK:HR:2023:1234]``.
3. If the context does not actually answer the question, say so plainly
   and stop. Do NOT invent citations or facts.
4. Surface any adverse-treatment caveats supplied under "Caveats" at the
   top of your answer (overruled / reversed / departed_from / criticised).
5. Always prepend the standing disclaimer the host system sends through.
6. Be concise. Mirror the user's jurisdiction and as-of date framing.

Citation prose conventions:
* US cases: use the reporter form when known (e.g. ``Brown v. Board, 347
  U.S. 483 (1954) [347 U.S. 483]``). Otherwise the bracketed id alone.
* UK / EW: use neutral-citation + section form (``Fraud Act 2006 s.2
  [uk/ukpga/2006/35/section/2@enacted]``).
* EU: case law → ECLI primary, CELEX secondary (``Keck Mithouard,
  ECLI:EU:C:1993:905 (CELEX 61991CJ0267) [ECLI:EU:C:1993:905]``).
  Legislation → short title + CELEX (``GDPR, Reg (EU) 2016/679,
  Art. 6 [eu/celex/32016R0679/article/art.6]``).
* DK: ECLI primary + parallel Ufr reporter when present (``U.2010.456H,
  ECLI:DK:HR:2010:456 [ECLI:DK:HR:2010:456]``); statutes by Danish short
  title + § (``Aftaleloven § 36 [dk/lbk/.../section/§36]``).
"""

# Civil-law framing for Danish-language answers: drop "binding precedent"
# language, use praksis / fast praksis. DA judges fravige rather than
# overrule. Mirror DA citation conventions throughout.
SYSTEM_PROMPT_DA = """Du er en forskningsassistent for spørgsmål om primær ret.

Strenge regler:
1. Brug UDELUKKENDE de passager, der findes under "Context" nedenfor.
2. Citer hvert faktuelt udsagn ved kildens identifikator i firkantede
   parenteser — præcis som den fremgår af konteksten, f.eks.
   ``[dk/lbk/2018/502/section/§6/stk.1]``, ``[ECLI:DK:HR:2023:1234]``
   eller ``[eu/celex/32016R0679/article/art.6]``.
3. Hvis konteksten ikke besvarer spørgsmålet, sig det klart og stop. Du
   må IKKE opfinde citater eller fakta.
4. Fremhæv eventuelle forbehold under "Caveats" øverst i dit svar
   (fraveget / kritiseret / tilsidesat).
5. Indled altid med den stående disclaimer, som systemet sender med.
6. Vær præcis. Spejl brugerens jurisdiktion og as-of dato.

Bemærk dansk retskildelære: Højesterets afgørelser udgør fast praksis
og har betydelig persuasiv vægt, men der er ikke binding precedent som
i common law. Tal om praksis, fast praksis, fravigelse fra praksis —
ikke om "overrule" eller "binding precedent".

Konventioner for citatformat:
* Danske domme: ECLI primært + parallel Ufr-henvisning når tilgængelig
  (``U.2010.456H, ECLI:DK:HR:2010:456 [ECLI:DK:HR:2010:456]``).
* Danske love: dansk kort titel + § (``Aftaleloven § 36
  [dk/lbk/.../section/§36]``; ``Straffelovens § 279 stk. 2
  [dk/lbk/.../section/§279/stk.2]``).
* EU-domme: ECLI primært + CELEX sekundært (``Keck Mithouard,
  ECLI:EU:C:1993:905 (CELEX 61991CJ0267) [ECLI:EU:C:1993:905]``).
* EU-lovgivning: kort titel + CELEX (``GDPR, Forordning (EU) 2016/679,
  Art. 6 [eu/celex/32016R0679/article/art.6]``).
"""


_SYSTEM_PROMPTS: dict[str, str] = {
    "en": SYSTEM_PROMPT_EN,
    "da": SYSTEM_PROMPT_DA,
}


def system_prompt_for(language: str | None) -> str:
    """Return the system prompt for ``language`` (EN fallback)."""
    if not language:
        return SYSTEM_PROMPT_EN
    return _SYSTEM_PROMPTS.get(language.lower(), SYSTEM_PROMPT_EN)


# --- citation format helpers ---------------------------------------------


# DK lbk Provision: dk/lbk/2018/502/section/§6/stk.1[/nr.1]
_DK_PROV_RE = re.compile(
    r"^dk/(?P<doc_type>[a-z]+)/(?P<year>\d{4})/(?P<num>\d+)/section/§(?P<sec>[^/]+)"
    r"(?:/stk\.(?P<stk>[^/]+))?(?:/nr\.(?P<nr>[^/]+))?$"
)

# EU Provision: eu/celex/32016R0679/article/art.6
_EU_PROV_RE = re.compile(
    r"^eu/celex/(?P<celex>[1-9]\d{4}[A-Z]{1,2}\d{4})/article/art\.(?P<art>.+)$"
)

# EU Instrument: eu/celex/32016R0679
_EU_INSTR_RE = re.compile(r"^eu/celex/(?P<celex>[1-9]\d{4}[A-Z]{1,2}\d{4})$")

# DK Instrument: dk/lbk/2018/502
_DK_INSTR_RE = re.compile(
    r"^dk/(?P<doc_type>[a-z]+)/(?P<year>\d{4})/(?P<num>\d+)$"
)

# UK Provision: uk/ukpga/2006/35/section/2@enacted (kept for cross-jurisdiction
# enrichment when an EN answer cites a UK statute).
_UK_PROV_RE = re.compile(
    r"^uk/(?P<act_type>[a-z]+)/(?P<year>\d{4})/(?P<num>\d+)/section/(?P<sec>[^@]+)@(?P<ver>.+)$"
)

# DK Ufr alt-id shape — Phase 1 cite_dk normalises to ``U.YYYY.NNNN.X``
# where ``X`` is H/V/Ø/B (Højesteret / Vestre / Østre / Byret). When a
# Case node carries this in ``Case.citations`` alongside its ECLI we
# surface both forms in the prompt heading (Phase 14.1).
_UFR_ALT_RE = re.compile(r"^U\.\d{4}\.\d{1,5}\.[HVØB]$")


def _pretty_ufr(normalised: str) -> str:
    """``U.2023.1234.H`` → ``U.2023.1234H`` (DA convention: no dot before court)."""
    return normalised[:-2] + normalised[-1]


def _pick_parallel_ufr(citations: Sequence[str] | None) -> str | None:
    """Return the first Ufr-shaped alt-id in ``citations``, prettified.

    Operators sometimes paste alt-ids in either the normalised
    ``U.YYYY.NNNN.X`` form (Phase 1 cite_dk output) or the surface
    ``U.YYYY.NNNNX`` form. Accept both.
    """
    if not citations:
        return None
    for c in citations:
        s = (c or "").strip()
        if _UFR_ALT_RE.match(s):
            return _pretty_ufr(s)
        # Surface form: U.2023.1234H — accept too.
        if re.match(r"^U\.\d{4}\.\d{1,5}[HVØB]$", s):
            return s
    return None


def _celex_form(celex: str) -> str:
    """``32016R0679`` → ``Reg (EU) 2016/679``; ``32019L0770`` → ``Dir (EU) 2019/770``."""
    if len(celex) < 9:
        return celex
    year = celex[1:5]
    type_char = celex[5]
    num = celex[6:].lstrip("0") or "0"
    kinds = {"R": "Reg (EU)", "L": "Dir (EU)", "D": "Dec (EU)", "H": "Rec (EU)"}
    kind = kinds.get(type_char, "")
    if kind:
        return f"{kind} {year}/{num}"
    return celex


def _format_dk_instrument_label(doc_type: str, year: str, num: str) -> str:
    """``lbk/2018/502`` → ``Lbk nr. 502 af 2018``."""
    return f"{doc_type.capitalize()} nr. {num} af {year}"


def _format_dk_provision_label(
    doc_type: str, year: str, num: str, sec: str, stk: str | None, nr: str | None
) -> str:
    """Human-readable DA section path: ``Lbk nr. 502 af 2018, § 6 stk. 1 nr. 1``."""
    base = _format_dk_instrument_label(doc_type, year, num)
    tail = f"§ {sec}"
    if stk:
        tail += f" stk. {stk}"
    if nr:
        tail += f" nr. {nr}"
    return f"{base}, {tail}"


def format_human_citation(
    parent_id: str,
    *,
    parent_type: str,
    parent_name: str = "",
    section_path: str | None = None,
    language: str = "en",
    citations: Sequence[str] | None = None,
) -> str:
    """Build a human-readable heading for the prompt context block.

    The bracketed ``[parent_id]`` is what the LLM must echo back — this
    function only enriches the surrounding prose so a reader sees a
    familiar form (``GDPR Art. 6``, ``Aftaleloven § 36``, …) instead of
    the raw slash-path twice.

    ``citations`` is the optional alt-id list from ``Case.citations``;
    when an ECLI:DK Case carries a parallel Ufr identifier
    (``U.YYYY.NNNN.X``) in it, the heading surfaces both forms — the DA
    convention DA lawyers actually read.

    Returns ``"<human prose> [parent_id]"`` or the bare id when no
    enrichment is recognised. Language affects DK / EU prose only; US /
    UK formats are identical across EN and DA outputs.
    """
    lang = (language or "en").lower()
    raw = f"[{parent_id}]"

    # EU Provision (article of a regulation / directive)
    m = _EU_PROV_RE.match(parent_id)
    if m is not None:
        celex_form = _celex_form(m["celex"])
        art = m["art"]
        prose = f"{celex_form}, Art. {art}" if lang == "en" else f"{celex_form}, Art. {art}"
        return f"{prose} {raw}"

    # EU Instrument
    m = _EU_INSTR_RE.match(parent_id)
    if m is not None:
        celex_form = _celex_form(m["celex"])
        return f"{celex_form} {raw}"

    # DK Provision
    m = _DK_PROV_RE.match(parent_id)
    if m is not None:
        # Prefer the human-friendly name if the candidate already has one
        # (e.g. ``Databeskyttelsesloven`` from the Instrument lookup).
        if parent_name and parent_name != parent_id:
            tail = section_path or f"§ {m['sec']}"
            if m["stk"]:
                tail += f" stk. {m['stk']}"
            if m["nr"]:
                tail += f" nr. {m['nr']}"
            return f"{parent_name}, {tail} {raw}"
        label = _format_dk_provision_label(
            m["doc_type"], m["year"], m["num"], m["sec"], m["stk"], m["nr"]
        )
        return f"{label} {raw}"

    # DK Instrument
    m = _DK_INSTR_RE.match(parent_id)
    if m is not None:
        if parent_name and parent_name != parent_id:
            return f"{parent_name} {raw}"
        label = _format_dk_instrument_label(m["doc_type"], m["year"], m["num"])
        return f"{label} {raw}"

    # ECLI:DK case — ECLI is canonical, parent_name is the caption when
    # present, parallel Ufr alt-id surfaces alongside when ``citations``
    # carries one (Phase 14.1 DA convention).
    if parent_id.startswith("ECLI:DK:"):
        ufr = _pick_parallel_ufr(citations)
        parts: list[str] = []
        if parent_name and parent_name != parent_id:
            parts.append(parent_name)
        if ufr:
            parts.append(ufr)
        parts.append(parent_id)
        return ", ".join(parts) + f" {raw}" if parts else raw

    # ECLI:EU case — ECLI is canonical, parent_name is the caption when present.
    if parent_id.startswith("ECLI:EU:"):
        if parent_name and parent_name != parent_id:
            return f"{parent_name}, {parent_id} {raw}"
        return raw

    # UK Provision — keep existing form for cross-jurisdiction EN answers.
    m = _UK_PROV_RE.match(parent_id)
    if m is not None:
        act_label = f"{m['act_type'].upper()} {m['year']} c.{m['num']}"
        section = f"s.{m['sec']}"
        return f"{act_label} {section} (version={m['ver']}) {raw}"

    # Case node with name — emulate the common-law caption form.
    if parent_type == "Case" and parent_name and parent_name != parent_id:
        return f"{parent_name} {raw}"

    # Fallback — raw id only.
    return raw


def format_candidates_block(
    candidates: Sequence["Candidate"], *, language: str = "en", max_chars: int = 1500
) -> str:
    """Serialise the reranked candidates into the prompt's Context block.

    Used by ``synthesize._context_block``. Each candidate gets a
    numbered heading via ``format_human_citation``, followed by its
    text body capped at ``max_chars``.
    """
    lines: list[str] = []
    for i, c in enumerate(candidates, 1):
        head = format_human_citation(
            c.parent_id,
            parent_type=c.parent_type,
            parent_name=c.parent_name,
            section_path=c.section_path,
            language=language,
            citations=c.extras.get("citations") if hasattr(c, "extras") else None,
        )
        body = (c.text or "").strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "…"
        lines.append(f"#{i} {head}\n{body}".strip())
    return "\n\n---\n\n".join(lines)


# --- empty-context messages -----------------------------------------------


_NO_CONTEXT_MESSAGES: dict[str, str] = {
    "en": (
        "No authorities matched the query. Retrieved context is empty; "
        "I won't speculate."
    ),
    "da": (
        "Ingen kilder svarede til forespørgslen. Den hentede kontekst er "
        "tom; jeg vil ikke spekulere."
    ),
}


def no_context_message_for(language: str | None) -> str:
    """Return the no-context fallback body for ``language`` (EN fallback)."""
    if not language:
        return _NO_CONTEXT_MESSAGES["en"]
    return _NO_CONTEXT_MESSAGES.get(language.lower(), _NO_CONTEXT_MESSAGES["en"])


__all__ = [
    "DISCLAIMER_EN",
    "DISCLAIMER_DA",
    "SYSTEM_PROMPT_EN",
    "SYSTEM_PROMPT_DA",
    "disclaimer_for",
    "system_prompt_for",
    "format_human_citation",
    "format_candidates_block",
    "no_context_message_for",
]
