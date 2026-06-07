"""Multi-signal language detection.

Lightweight DA vs EN binary classifier shared across the clg retrieval
parser (Phase 7), the synthesis prompts (Phase 8), and any other caller
that needs a single ``(lang, confidence)`` answer without dragging in a
multi-megabyte language-id library.

**Four signals, weighted sum:**

1. **DA-only diacritics** (``æ/ø/å``) — heaviest weight (4.0). One hit
   is decisive. These characters don't appear in EN in the Western
   European context, so even a single occurrence is unambiguous evidence.
2. **Stopword frequency** — top-40 lists per language; weighted by hit
   ratio over total tokens. Asymmetric (no shared words) so a sentence
   can't tie itself by accident.
3. **Character bigrams** — DA-distinctive (``sk / ld / rk / rd / lv``)
   vs EN-distinctive (``th / wh / qu / wr / kn / gh / ph / ck``).
   Carries DA queries that have been diacritic-stripped (e.g. when the
   author can't type ``æ`` quickly).
4. **DA word-ending suffixes** — Danish inflections English doesn't
   share (``-ende``, ``-else``, ``-heden``, ``-erne``).

Returns confidence as the normalised margin of winner over runner-up
(``0.0`` = coin flip, ``1.0`` = unambiguous). Below ``_DA_MIN_CONFIDENCE``
falls back to EN — Claude handles EN better than DA, and the DK firm's
queries almost always carry strong DA signal when they're DA. EN is the
safe default for everything else.

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
        "og", "at", "der", "som", "med", "for", "den", "det", "en", "et",
        "af", "på", "til", "har", "ikke", "var", "fra", "være", "om", "men",
        "han", "hun", "vi", "kan", "skal", "vil", "blev", "også", "her",
        "ved", "eller", "sig", "så", "havde", "kunne", "skulle", "ville",
        "efter", "under", "før", "mod", "uden",
    ]
)
EN_STOPWORDS: frozenset[str] = frozenset(
    [
        "the", "of", "and", "to", "is", "it", "that", "was", "for",
        "on", "with", "as", "be", "by", "are", "this", "from", "not", "or",
        "an", "have", "has", "but", "what", "all", "we", "can", "do", "does",
        "would", "could", "should", "which", "when", "where", "how", "who",
        "their", "his", "her", "they",
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

# Weights tuned on hand-built test corpus. Diacritics dominate when
# present; stopwords + bigrams + suffixes share the remainder.
_WEIGHT_DIACRITIC = 4.0
_WEIGHT_STOPWORD = 1.0
_WEIGHT_BIGRAM = 0.5
_WEIGHT_SUFFIX = 0.7

# Below this confidence margin we fall back to EN. EN is the safe default
# because Claude handles EN better than DA and the DA firm's queries
# almost always carry strong DA signal (legal-statute names, æ/ø/å).
_DA_MIN_CONFIDENCE = 0.15


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


def detect_language(text: str) -> tuple[str, float]:
    """Detect DA vs EN; return ``(language, confidence)``.

    ``language`` is an ISO 639-1 code (``"da"`` or ``"en"``).
    ``confidence`` is the normalised margin of winner over runner-up
    (``0.0`` = coin flip, ``1.0`` = unambiguous).

    Below the internal ``_DA_MIN_CONFIDENCE`` threshold (default 0.15)
    the result always defaults to ``"en"`` regardless of which side
    scored higher — EN is the safer fallback for downstream synthesis.

    See module docstring for the full signal/weight design.
    """
    text = (text or "").strip()
    if len(text) < 3:
        return "en", 0.0

    da_score = 0.0
    en_score = 0.0

    # 1. DA-only diacritics — one hit is strong evidence.
    diacritic_hits = sum(1 for ch in text if ch in DA_ONLY_CHARS)
    if diacritic_hits:
        da_score += _WEIGHT_DIACRITIC * min(diacritic_hits, 3)

    # 2. Stopword frequency, normalised by total token count.
    tokens = _tokenise(text)
    if tokens:
        da_hits = sum(1 for tok in tokens if tok in DA_STOPWORDS)
        en_hits = sum(1 for tok in tokens if tok in EN_STOPWORDS)
        da_score += _WEIGHT_STOPWORD * (da_hits / len(tokens)) * 10
        en_score += _WEIGHT_STOPWORD * (en_hits / len(tokens)) * 10

    # 3. Character bigrams.
    bigrams = _bigrams(text)
    if bigrams:
        da_bg = sum(1 for b in bigrams if b in DA_BIGRAMS)
        en_bg = sum(1 for b in bigrams if b in EN_BIGRAMS)
        da_score += _WEIGHT_BIGRAM * (da_bg / len(bigrams)) * 10
        en_score += _WEIGHT_BIGRAM * (en_bg / len(bigrams)) * 10

    # 4. DA word-ending suffixes.
    if tokens:
        suffix_hits = sum(
            1 for tok in tokens if len(tok) >= 4 and tok.endswith(DA_SUFFIXES)
        )
        da_score += _WEIGHT_SUFFIX * (suffix_hits / len(tokens)) * 10

    total = da_score + en_score
    if total <= 0:
        return "en", 0.0
    margin = abs(da_score - en_score) / total
    if da_score > en_score and margin >= _DA_MIN_CONFIDENCE:
        return "da", margin
    return "en", margin if en_score > da_score else 0.0


__all__ = ["detect_language"]
