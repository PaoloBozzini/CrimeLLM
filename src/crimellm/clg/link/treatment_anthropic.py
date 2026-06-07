"""Tier 4 — Claude. Escalation only AND the teacher for distillation.

Two roles:

1. **Bulk escalation** — the last 2-5% of edges the cheaper tiers can't
   confidently label. Use ``ClaudeBatchTreatmentClassifier`` here; it
   relies on the Anthropic SDK's prompt caching so the system prompt +
   vocab description are billed once per cache window, not per request.

2. **Teacher for the distilled head** — sample N edges, label them with
   Claude, write a CSV in ``train.py``-compatible shape. Phase 5.2.

Both roles use the same prompt + schema-shaped JSON output so we can audit
agreement between the two paths.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence

from .treatment_base import (
    TREATMENT_VOCAB,
    EdgeContext,
    TreatmentClassifier,
    TreatmentResult,
)

_SYSTEM_PROMPT = (
    "You classify how a citing legal opinion treats a cited case.\n\n"
    f"Allowed labels (controlled vocabulary): {', '.join(TREATMENT_VOCAB)}.\n\n"
    "Output ONLY a JSON object on a single line: "
    '{"label": "<one of the above>", "confidence": <0..1>, "reason": "<≤120 chars>"}.\n'
    "No code fences. No prose outside the JSON. When the citing sentence is "
    'unavailable or genuinely ambiguous, return {"label": "neutral", "confidence": 0.3, ...}.'
)


def _user_prompt(edge: EdgeContext) -> str:
    return (
        f"Citing case: {edge.citing_case_name or edge.citing_case_id}\n"
        f"Cited case: {edge.cited_case_name or edge.cited_case_id}\n"
        f"Citing sentence: {edge.citing_sentence or '(unavailable)'}\n"
    )


class ClaudeTreatmentClassifier(TreatmentClassifier):
    """Synchronous Claude classifier. Use for small per-edge calls.

    Uses prompt caching on the system block so repeat calls reuse the
    vocabulary description. For bulk teacher labelling (5.2) or escalation
    over a long tail of edges, use the streaming/batch variant instead.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 200,
        api_key: str | None = None,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "anthropic package not installed. Add the [anthropic] or [clg] extra."
            ) from e

        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; cannot use ClaudeTreatmentClassifier."
            )

        self.model = model
        self._max_tokens = max_tokens
        self._client = Anthropic(api_key=key)

    def classify_batch(self, edges: Sequence[EdgeContext]) -> list[TreatmentResult | None]:
        out: list[TreatmentResult | None] = []
        for edge in edges:
            out.append(self._call_one(edge))
        return out

    def _call_one(self, edge: EdgeContext) -> TreatmentResult:
        start = time.perf_counter()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": _user_prompt(edge)}],
        )
        raw = "".join(getattr(p, "text", "") for p in msg.content).strip()
        latency = (time.perf_counter() - start) * 1000.0
        try:
            data = json.loads(raw)
            label = str(data.get("label", "")).lower()
            conf = float(data.get("confidence", 0.0))
            if label not in TREATMENT_VOCAB:
                raise ValueError(f"label {label!r} not in vocab")
            return TreatmentResult(
                label=label,  # type: ignore[arg-type]
                confidence=max(0.0, min(1.0, conf)),
                source=f"{self.name}:{self.model}",
                latency_ms=latency,
                extras={
                    "reason": data.get("reason", "")[:200],
                    "cache_creation_input_tokens": getattr(
                        msg.usage, "cache_creation_input_tokens", 0
                    ),
                    "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0),
                },
            )
        except Exception as e:  # noqa: BLE001
            return TreatmentResult(
                label="neutral",
                confidence=0.0,
                source=f"{self.name}:parse-error",
                latency_ms=latency,
                extras={"error": f"{type(e).__name__}: {e}", "raw": raw[:200]},
            )
