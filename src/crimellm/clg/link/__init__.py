"""Citation extraction (eyecite) + treatment classification (cascade).

Tier 1: rules (``treatment_rules``) — fast, free, ~35% coverage.
Tier 2: distilled head (``treatment_distilled``) — needs Phase 5.2 model.
Tier 3: local LLM via Ollama (``treatment_local_llm``).
Tier 4: Claude Haiku with prompt caching (``treatment_anthropic``).
Orchestrator: ``treatment_cascade.CascadeClassifier`` with per-tier thresholds.
"""

from .citation_context import extract_citing_sentence
from .treatment_anthropic import ClaudeTreatmentClassifier
from .treatment_base import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    TREATMENT_VOCAB,
    EdgeContext,
    TreatmentClassifier,
    TreatmentLabel,
    TreatmentResult,
)
from .treatment_cascade import CascadeClassifier, CascadeReport, CascadeTelemetry
from .treatment_distilled import DistilledTreatmentClassifier
from .treatment_local_llm import OllamaTreatmentClassifier
from .treatment_rules import RuleTreatmentClassifier

__all__ = [
    "EdgeContext",
    "TreatmentResult",
    "TreatmentClassifier",
    "TreatmentLabel",
    "TREATMENT_VOCAB",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "RuleTreatmentClassifier",
    "DistilledTreatmentClassifier",
    "OllamaTreatmentClassifier",
    "ClaudeTreatmentClassifier",
    "CascadeClassifier",
    "CascadeReport",
    "CascadeTelemetry",
    "extract_citing_sentence",
]
