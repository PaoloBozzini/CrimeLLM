"""Direct tests for :mod:`crimellm.common.language`.

The clg retrieval tests already cover the user-facing parse_query
integration; these tests focus on the detector itself so callers from
other modules (synthesis, ingest, future stacks) can rely on a stable
contract independent of the clg pipeline.
"""

from __future__ import annotations

import pytest

from crimellm.common import detect_language as detect_language_reexport
from crimellm.common.language import (
    DA_BIGRAMS,
    DA_ONLY_CHARS,
    DA_STOPWORDS,
    DA_SUFFIXES,
    EN_BIGRAMS,
    EN_STOPWORDS,
    detect_language,
)


# --- public surface -------------------------------------------------------


def test_reexported_from_common_package():
    assert detect_language_reexport is detect_language


def test_signal_tables_are_frozensets():
    """Mutability would let a downstream caller silently corrupt the
    detector for every other caller in the process."""
    assert isinstance(DA_ONLY_CHARS, frozenset)
    assert isinstance(DA_STOPWORDS, frozenset)
    assert isinstance(EN_STOPWORDS, frozenset)
    assert isinstance(DA_BIGRAMS, frozenset)
    assert isinstance(EN_BIGRAMS, frozenset)


def test_stopword_lists_disjoint():
    """Asymmetric signal — a token must score for at most one language."""
    assert DA_STOPWORDS.isdisjoint(EN_STOPWORDS)


def test_bigram_lists_disjoint():
    assert DA_BIGRAMS.isdisjoint(EN_BIGRAMS)


def test_suffix_tuple_no_duplicates():
    """Duplicate suffixes inflate the score asymmetrically."""
    assert len(DA_SUFFIXES) == len(set(DA_SUFFIXES))


# --- return shape --------------------------------------------------------


def test_return_shape():
    lang, conf = detect_language("anything")
    assert isinstance(lang, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_empty_input_defaults_to_en():
    assert detect_language("")[0] == "en"
    assert detect_language("   ")[0] == "en"


def test_too_short_input_defaults_to_en():
    assert detect_language("hi")[0] == "en"
    assert detect_language("ab")[0] == "en"


def test_returns_iso_639_1_codes_only():
    """Contract: caller should never see 'und' or anything weird."""
    for text in (
        "",
        "ja",
        "Højesteret",
        "the Court held",
        "12345 !!! ???",
        "æøå",
    ):
        lang, _ = detect_language(text)
        assert lang in {"da", "en"}


# --- DA detection --------------------------------------------------------


def test_diacritics_lock_in_da():
    lang, conf = detect_language("Højesteret har afsagt en afgørelse.")
    assert lang == "da"
    assert conf >= 0.3


def test_da_without_diacritics_via_stopwords_and_suffixes():
    """Diacritic-stripped DA query should still resolve to DA via the
    other three signals."""
    lang, _ = detect_language(
        "Hvilke regler galder for behandling af persondata efter loven?"
    )
    assert lang == "da"


def test_da_via_bigrams_alone():
    """Compact DA-flavoured text with no stopwords still picks DA via
    distinctive bigrams + diacritics."""
    lang, _ = detect_language("Højesteret afsagde dom om straffelovens § 279.")
    assert lang == "da"


@pytest.mark.parametrize(
    "text",
    [
        "Højesteret har truffet en afgørelse om aftalelovens § 36.",
        "Loven gælder for behandling af personoplysninger.",
        "Sagen er behandlet under henvisning til U.2010.456H.",
        "Tiltalt for overtrædelse af straffelovens § 279, stk. 2.",
        "Klager over Datatilsynets afgørelser kan ikke indbringes.",
    ],
)
def test_real_da_legal_queries(text):
    lang, conf = detect_language(text)
    assert lang == "da"
    assert conf >= 0.2


# --- EN detection --------------------------------------------------------


def test_pure_english_returns_en():
    lang, conf = detect_language(
        "What does the Fraud Act 2006 say about phishing?"
    )
    assert lang == "en"
    assert conf > 0.0


def test_en_with_da_term_stays_en():
    """The bulk of an EN sentence carries even when a DA statute name is
    embedded mid-clause."""
    lang, _ = detect_language(
        "What does the Danish statute straffelovens § 279 say about fraud?"
    )
    assert lang == "en"


@pytest.mark.parametrize(
    "text",
    [
        "Has the CJEU interpreted GDPR Article 6(1)(f) on marketing?",
        "Does the Theft Act 1968 cover digital theft?",
        "What is the rule from Brown v. Board of Education?",
        "When was Plessy v. Ferguson overruled?",
        "How does Article 101 TFEU apply to vertical agreements?",
    ],
)
def test_real_en_legal_queries(text):
    lang, _ = detect_language(text)
    assert lang == "en"


# --- robustness ----------------------------------------------------------


def test_punctuation_only_falls_back_to_en():
    assert detect_language("!?!?@#$%^")[0] == "en"


def test_unicode_safe():
    # No crashes on emoji, CJK, etc. EN fallback expected.
    for text in ("🚀🚀🚀", "中文测试", "test 中 mixed"):
        lang, _ = detect_language(text)
        assert lang in {"da", "en"}


def test_case_insensitivity():
    """Upper-case text should detect the same as lower-case."""
    upper = detect_language("HØJESTERET HAR AFSAGT EN AFGØRELSE.")
    lower = detect_language("højesteret har afsagt en afgørelse.")
    assert upper[0] == lower[0] == "da"


def test_confidence_monotonic_with_evidence():
    """More DA signal → higher DA confidence."""
    one_signal = detect_language("Højesteret")[1]
    many_signals = detect_language(
        "Højesteret afsagde dom om straffelovens § 279 med henvisning til afgørelsen."
    )[1]
    assert many_signals >= one_signal
