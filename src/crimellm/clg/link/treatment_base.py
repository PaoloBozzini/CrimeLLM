"""Treatment classification — shared types.

All backends take an ``EdgeContext`` (the citing case, the cited case, and
the surrounding sentence in the citing opinion) and return a
``TreatmentResult`` (one of the 10 vocabulary labels + a confidence score
+ the backend name + telemetry).

Keep this module dependency-light so importing the cascade doesn't drag in
the ML stack or the Anthropic SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

# Mirrors clg.models.Treatment but redefined here so a fresh import path
# doesn't trip over the dataclass module. Common-law labels first, then the
# civil-law set (DK Højesteret-style departures).
TreatmentLabel = Literal[
    "followed",
    "applied",
    "considered",
    "distinguished",
    "doubted",
    "not_followed",
    "overruled",
    "reversed",
    "affirmed",
    "neutral",
    "departed_from",
    "criticised",
]

TREATMENT_VOCAB: tuple[TreatmentLabel, ...] = (
    "followed",
    "applied",
    "considered",
    "distinguished",
    "doubted",
    "not_followed",
    "overruled",
    "reversed",
    "affirmed",
    "neutral",
    "departed_from",
    "criticised",
)

LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(TREATMENT_VOCAB)}
ID_TO_LABEL: dict[int, str] = {i: label for label, i in LABEL_TO_ID.items()}


@dataclass(slots=True)
class EdgeContext:
    """One ``(citing_case)-[:CITES]->(cited_case)`` row to classify.

    ``citing_sentence`` is the high-signal field. When the upstream extractor
    couldn't isolate one — typical for older CourtListener opinions where
    eyecite mis-segments — we still call the classifier but the rule and
    distilled tiers will likely abstain and the LLM tiers will get less to
    work with.
    """

    citing_case_id: str
    cited_case_id: str
    citing_sentence: str = ""
    citing_case_name: str = ""
    cited_case_name: str = ""
    citing_decision_date: str | None = None
    cited_decision_date: str | None = None
    depth: float = 1.0


@dataclass(slots=True)
class TreatmentResult:
    """Output of a single classifier tier."""

    label: TreatmentLabel
    confidence: float
    source: str
    latency_ms: float = 0.0
    extras: dict[str, object] = field(default_factory=dict)


class TreatmentClassifier(ABC):
    """Sequence-aware classifier. ``classify_batch`` is the workhorse.

    Tiers may return ``None`` from a slot in the batch to mean "I abstain"
    — the cascade orchestrator interprets that as "send this edge to the
    next tier". A label with low confidence is *not* the same as abstaining;
    confidence is the cascade's escalation signal.
    """

    name: str

    @abstractmethod
    def classify_batch(self, edges: Sequence[EdgeContext]) -> list[TreatmentResult | None]:
        """Return one result per input, in order. ``None`` = abstain."""

    def classify(self, edge: EdgeContext) -> TreatmentResult | None:
        return self.classify_batch([edge])[0]


# --- batching helper -------------------------------------------------------


def in_batches(items: Iterable[EdgeContext], batch_size: int = 64) -> Iterable[list[EdgeContext]]:
    buf: list[EdgeContext] = []
    for it in items:
        buf.append(it)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf
