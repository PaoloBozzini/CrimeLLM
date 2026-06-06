"""Tier 3 — local LLM via Ollama with JSON-schema-constrained output.

Same Ollama runtime as ``OllamaSynthesizer``. Different prompt + a strict
output schema so we get a structured ``{label, confidence}`` back instead
of free-text reasoning.

Tradeoffs vs the distilled head:

* Slower per edge (~10-50/s on a Mac vs hundreds/s for the encoder).
* No training data needed.
* Easier to swap models — ``--model qwen2.5:14b-instruct`` etc. with one CLI flag.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from .treatment_base import (
    TREATMENT_VOCAB,
    EdgeContext,
    TreatmentClassifier,
    TreatmentResult,
)

_SYSTEM = (
    "You classify how a citing legal opinion treats a cited case. "
    "Output ONLY a JSON object matching the schema. No prose. No code fences."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(TREATMENT_VOCAB)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}


def _user_prompt(edge: EdgeContext) -> str:
    return (
        "Citing case: "
        f"{edge.citing_case_name or edge.citing_case_id}\n"
        "Cited case: "
        f"{edge.cited_case_name or edge.cited_case_id}\n"
        f"Citing sentence: {edge.citing_sentence or '(unavailable)'}\n\n"
        "Classify the treatment using the controlled vocabulary "
        f"({', '.join(TREATMENT_VOCAB)}). When the sentence is unavailable "
        'or ambiguous, return {"label": "neutral", "confidence": 0.3}.'
    )


class OllamaTreatmentClassifier(TreatmentClassifier):
    """Tier 3: Ollama-served local LLM with JSON-schema-constrained output."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def classify_batch(self, edges: Sequence[EdgeContext]) -> list[TreatmentResult | None]:
        import httpx

        out: list[TreatmentResult | None] = []
        with httpx.Client(timeout=self.timeout) as client:
            for edge in edges:
                start = time.perf_counter()
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": _user_prompt(edge)},
                    ],
                    "stream": False,
                    "format": _SCHEMA,
                    "options": {"temperature": self.temperature},
                }
                try:
                    r = client.post(f"{self.host}/api/chat", json=payload)
                    r.raise_for_status()
                    raw = r.json().get("message", {}).get("content", "") or ""
                    data = json.loads(raw)
                    label = str(data.get("label", "")).lower()
                    conf = float(data.get("confidence", 0.0))
                    if label not in TREATMENT_VOCAB:
                        out.append(None)
                        continue
                    out.append(
                        TreatmentResult(
                            label=label,  # type: ignore[arg-type]
                            confidence=max(0.0, min(1.0, conf)),
                            source=f"{self.name}:{self.model}",
                            latency_ms=(time.perf_counter() - start) * 1000.0,
                            extras={"raw": raw[:200]},
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    out.append(
                        TreatmentResult(
                            label="neutral",
                            confidence=0.0,
                            source=f"{self.name}:error",
                            latency_ms=(time.perf_counter() - start) * 1000.0,
                            extras={"error": f"{type(e).__name__}: {e}"},
                        )
                    )
        return out
