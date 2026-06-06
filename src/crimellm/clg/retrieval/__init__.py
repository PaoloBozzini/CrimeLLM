"""seed -> traverse -> good-law -> rerank -> synthesize. Phase 4."""

from .expand import expand_seeds
from .good_law import GoodLawFlag, check_good_law, summary_label
from .parse_query import Query, parse_query
from .query import run_query
from .rerank import RerankWeights, dedupe_candidates, rerank
from .seed import Candidate, seed_from_chunks
from .synthesize import (
    AirLLMSynthesizer,
    Answer,
    AnthropicSynthesizer,
    FakeSynthesizer,
    OllamaSynthesizer,
    Synthesizer,
    check_citations,
    extract_citations,
    get_synthesizer,
)

__all__ = [
    "Query",
    "parse_query",
    "Candidate",
    "seed_from_chunks",
    "expand_seeds",
    "GoodLawFlag",
    "check_good_law",
    "summary_label",
    "RerankWeights",
    "dedupe_candidates",
    "rerank",
    "Answer",
    "Synthesizer",
    "AnthropicSynthesizer",
    "OllamaSynthesizer",
    "AirLLMSynthesizer",
    "FakeSynthesizer",
    "get_synthesizer",
    "extract_citations",
    "check_citations",
    "run_query",
]
