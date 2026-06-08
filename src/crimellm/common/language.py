"""Multi-signal language detection.

Lightweight 4-way classifier (DA / EN / FR / DE) shared across the clg
retrieval parser (Phase 7), the synthesis prompts (Phase 8), and any
other caller that needs a single ``(lang, confidence)`` answer without
dragging in a multi-megabyte language-id library.

**Signals (weighted sum, per language):**

1. **Language-specific diacritics / characters** — heaviest weight
   (4.0). One hit per language is decisive:
   - DA: ``æ/ø/å`` (don't appear in EN/FR/DE in Western European text)
   - FR: ``ç/œ`` (œ is uniquely FR; ç also Portuguese but rare in our
     legal corpus)
   - DE: ``ß`` (uniquely German; eszett doesn't appear elsewhere)
   - EN: no language-only diacritics — earns its score from other signals
2. **Stopword frequency** — top stopword lists per language, weighted by
   hit ratio over total tokens. Disjoint across languages (Phase 7
   asymmetry invariant) so a sentence can't tie itself by accident.
3. **Character bigrams** — language-distinctive pairs
   (``th`` for EN, ``sk`` for DA, ``ou`` for FR, ``sch`` trigram for DE
   captured as ``sc`` + ``ch`` bigrams).
4. **Word-ending suffixes** — language-specific inflections
   (DA ``-ende``, FR ``-tion -ment -aux``, DE ``-ung -keit -lich``).

Returns confidence as the normalised margin of winner over runner-up
(``0.0`` = coin flip, ``1.0`` = unambiguous). Below the per-language
``_MIN_CONFIDENCE`` (default 0.15) falls back to **EN** — Claude handles
EN better than DA/FR/DE, and queries with strong non-EN signal almost
always carry decisive diacritics or stopword density. EN is the safe
default for everything else.

**Drop-in upgrade path:** for short-text accuracy beyond what hand-tuned
heuristics give, swap in ``langdetect`` (CLD2 port) or ``langid.py`` —
both return ISO 639-1 codes and a confidence; wrap them in a thin shim
that preserves this module's ``(lang, confidence)`` return contract.

Pure stdlib. No new dependencies.
"""

from __future__ import annotations

# Strong DA signal: characters that don't appear in EN. One occurrence
# pushes the score hard toward DA.
DA_ONLY_CHARS = frozenset("æøåÆØÅ")

# Top-frequency stopwords. Curated to avoid overlap so the signal stays
# asymmetric: a word that's shared (e.g. ``i`` exists in both English
# "I" and Danish "i") gets dropped from at least one list.
DA_STOPWORDS: frozenset[str] = frozenset(
    [
        "og", "at", "der", "som", "med", "den", "det", "en", "et",
        "af", "på", "til", "har", "ikke", "var", "fra", "være", "om", "men",
        "han", "hun", "vi", "kan", "skal", "vil", "blev", "også",
        "ved", "eller", "sig", "så", "havde", "kunne", "skulle", "ville",
        "efter", "under", "før", "mod", "uden",
    ]
)
EN_STOPWORDS: frozenset[str] = frozenset(
    [
        "the", "of", "and", "to", "is", "it", "that", "was",
        "on", "with", "as", "be", "by", "are", "this", "from", "not", "or",
        "an", "have", "has", "but", "what", "all", "we", "can", "do", "does",
        "would", "could", "should", "which", "when", "where", "how", "who",
        "their", "his", "they",
    ]
)

# Character bigrams that are markedly more frequent in one language than
# the other. Counted with simple overlap windowing on the lowercased text.
DA_BIGRAMS: frozenset[str] = frozenset(
    ["sk", "ld", "rk", "rd", "lv", "ev", "ud", "rl", "kk", "lg", "gn", "dt"]
)
EN_BIGRAMS: frozenset[str] = frozenset(
    ["th", "wh", "qu", "wr", "kn", "gh", "ph", "ck"]
)

# DA noun / verb inflections that English doesn't share. Counted on
# whitespace-tokenised words by suffix match; minimum length 4 avoids
# false hits on short EN words ("ben"/"den" etc.).
DA_SUFFIXES: tuple[str, ...] = ("ende", "else", "heden", "erne", "et", "en", "er")


# --- FR signals -----------------------------------------------------------

# œ is uniquely French. ç also appears in Portuguese but rare in our
# legal corpus — treating it as FR signal is acceptable noise.
FR_ONLY_CHARS = frozenset("çœÇŒ")

FR_STOPWORDS: frozenset[str] = frozenset(
    [
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou",
        "que", "qui", "dans", "pour", "par", "sur", "avec", "sans", "sous",
        "ne", "pas", "ces", "cette", "ce", "cet", "aux", "au", "ses",
        "son", "sa", "leur", "leurs", "est", "sont", "était", "été",
        "doit", "peut", "selon", "alors", "lors", "ainsi", "donc",
    ]
)

FR_BIGRAMS: frozenset[str] = frozenset(
    ["qu", "ou", "ai", "eu", "oi", "tr", "br", "ch", "ll", "ée"]
)

FR_SUFFIXES: tuple[str, ...] = ("tion", "ment", "ique", "able", "aux", "ées", "iste")


# --- DE signals -----------------------------------------------------------

# ß is uniquely German. Umlauts (ä/ö/ü) appear in DA too — keep them out
# of the "only" set and rely on stopwords + suffixes for disambiguation.
DE_ONLY_CHARS = frozenset("ß")

DE_STOPWORDS: frozenset[str] = frozenset(
    [
        "der", "die", "das", "und", "ist", "nicht", "ein", "eine", "auf",
        "von", "mit", "im", "zu", "auch", "wenn", "als", "aber", "auch",
        "dass", "wird", "war", "sein", "kann", "muss", "soll", "nach",
        "über", "unter", "vor", "wegen", "durch", "ohne", "gegen",
        "zwischen", "sowie", "bzw", "diese", "dieser", "dieses", "jeder",
    ]
)

DE_BIGRAMS: frozenset[str] = frozenset(
    ["sc", "ch", "ie", "ei", "au", "eu", "tz", "pf", "nd", "ng"]
)

DE_SUFFIXES: tuple[str, ...] = ("ung", "keit", "heit", "lich", "schaft", "lung", "isch")


# Weights tuned on hand-built test corpus. Diacritics dominate when
# present; stopwords + bigrams + suffixes share the remainder.
_WEIGHT_DIACRITIC = 4.0
_WEIGHT_STOPWORD = 1.0
_WEIGHT_BIGRAM = 0.5
_WEIGHT_SUFFIX = 0.7

# Below this confidence margin we fall back to EN. EN is the safe default
# because Claude handles EN best and non-EN queries with real signal
# almost always carry decisive diacritics or stopword density.
_MIN_CONFIDENCE = 0.15
_DA_MIN_CONFIDENCE = _MIN_CONFIDENCE  # back-compat alias


def _tokenise(text: str) -> list[str]:
    """Whitespace + punctuation split. Cheap, deterministic, regex-free."""
    out: list[str] = []
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalpha() or ch in "æøå":
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def _bigrams(text: str) -> list[str]:
    s = text.lower()
    return [s[i : i + 2] for i in range(len(s) - 1)]


def _tokenise_multi(text: str) -> list[str]:
    """Like ``_tokenise`` but keeps æ/ø/å/ç/œ/ß/ä/ö/ü inside tokens."""
    out: list[str] = []
    buf: list[str] = []
    extra = "æøåçœßäöü"
    for ch in text.lower():
        if ch.isalpha() or ch in extra:
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def _score_language(
    text: str,
    *,
    diacritics: frozenset[str],
    stopwords: frozenset[str],
    bigrams_set: frozenset[str],
    suffixes: tuple[str, ...],
    tokens: list[str],
    text_bigrams: list[str],
) -> float:
    """Compute the weighted score for one language."""
    score = 0.0
    if diacritics:
        hits = sum(1 for ch in text if ch in diacritics)
        if hits:
            score += _WEIGHT_DIACRITIC * min(hits, 3)
    if tokens:
        sw_hits = sum(1 for tok in tokens if tok in stopwords)
        score += _WEIGHT_STOPWORD * (sw_hits / len(tokens)) * 10
    if text_bigrams:
        bg_hits = sum(1 for b in text_bigrams if b in bigrams_set)
        score += _WEIGHT_BIGRAM * (bg_hits / len(text_bigrams)) * 10
    if tokens and suffixes:
        suf_hits = sum(
            1 for tok in tokens if len(tok) >= 4 and tok.endswith(suffixes)
        )
        score += _WEIGHT_SUFFIX * (suf_hits / len(tokens)) * 10
    return score


def detect_language(text: str) -> tuple[str, float]:
    """Detect DA / EN / FR / DE; return ``(language, confidence)``.

    ``language`` is an ISO 639-1 code (``"da"`` / ``"en"`` / ``"fr"`` /
    ``"de"``). ``confidence`` is the normalised margin of winner over
    runner-up (``0.0`` = coin flip, ``1.0`` = unambiguous).

    Below the internal ``_MIN_CONFIDENCE`` threshold (default 0.15) the
    result always defaults to ``"en"`` regardless of which side scored
    higher — EN is the safer fallback for downstream synthesis.

    See module docstring for the full signal/weight design.
    """
    text = (text or "").strip()
    if len(text) < 3:
        return "en", 0.0

    tokens = _tokenise_multi(text)
    text_bigrams = _bigrams(text)

    scores: dict[str, float] = {
        "da": _score_language(
            text,
            diacritics=DA_ONLY_CHARS,
            stopwords=DA_STOPWORDS,
            bigrams_set=DA_BIGRAMS,
            suffixes=DA_SUFFIXES,
            tokens=tokens,
            text_bigrams=text_bigrams,
        ),
        "en": _score_language(
            text,
            diacritics=frozenset(),  # no EN-only diacritics
            stopwords=EN_STOPWORDS,
            bigrams_set=EN_BIGRAMS,
            suffixes=(),  # EN suffixes too varied to score reliably
            tokens=tokens,
            text_bigrams=text_bigrams,
        ),
        "fr": _score_language(
            text,
            diacritics=FR_ONLY_CHARS,
            stopwords=FR_STOPWORDS,
            bigrams_set=FR_BIGRAMS,
            suffixes=FR_SUFFIXES,
            tokens=tokens,
            text_bigrams=text_bigrams,
        ),
        "de": _score_language(
            text,
            diacritics=DE_ONLY_CHARS,
            stopwords=DE_STOPWORDS,
            bigrams_set=DE_BIGRAMS,
            suffixes=DE_SUFFIXES,
            tokens=tokens,
            text_bigrams=text_bigrams,
        ),
    }

    total = sum(scores.values())
    if total <= 0:
        return "en", 0.0

    # Winner = argmax; confidence = margin of winner over best non-winner
    # normalised by total. Below threshold → EN fallback.
    winner = max(scores, key=lambda k: scores[k])
    runner_up = max((v for k, v in scores.items() if k != winner), default=0.0)
    margin = (scores[winner] - runner_up) / total

    if winner == "en":
        return "en", margin
    if margin >= _MIN_CONFIDENCE:
        return winner, margin
    return "en", margin


__all__ = ["detect_language"]
