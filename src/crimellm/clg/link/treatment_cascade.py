"""Cascade orchestrator. Threshold-based escalation across tiers.

Each tier runs in order. For each edge:

* The tier returns a ``TreatmentResult`` with ``confidence >= threshold`` →
  edge is finalised, no further tiers consulted.
* The tier returns ``None`` (abstain) → escalate.
* The tier returns a low-confidence result → escalate, **but remember it**
  as a fallback in case every downstream tier also fails.

After the cascade, every edge has either an accepted result or — at worst —
the best-effort fallback from whatever tier looked at it. The orchestrator
also emits per-edge telemetry (which tier accepted, confidence, latency)
so we can audit how often each tier carries its weight.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from .treatment_base import EdgeContext, TreatmentClassifier, TreatmentResult


@dataclass(slots=True)
class CascadeTelemetry:
    edge_index: int
    accepted_tier: str
    accepted_confidence: float
    fallback_used: bool
    tier_attempts: list[str]
    total_latency_ms: float


@dataclass(slots=True)
class CascadeReport:
    results: list[TreatmentResult]
    telemetry: list[CascadeTelemetry]

    def by_tier(self) -> dict[str, int]:
        """Count of edges accepted by each tier — handy for after-action reports."""
        out: dict[str, int] = {}
        for t in self.telemetry:
            out[t.accepted_tier] = out.get(t.accepted_tier, 0) + 1
        return out


class CascadeClassifier:
    """Pipeline of ``TreatmentClassifier`` tiers + per-tier confidence thresholds."""

    def __init__(
        self,
        tiers: Sequence[tuple[TreatmentClassifier, float]],
        *,
        budget_per_tier: dict[str, int] | None = None,
    ):
        """
        Args:
            tiers: ordered ``[(classifier, confidence_threshold), ...]``. Cheap first.
            budget_per_tier: max number of times each tier may be invoked in a
                single ``classify`` call. Use to cap the expensive escalation
                tier (e.g. ``{"anthropic": 100_000}``). Untracked tiers run unlimited.
        """
        if not tiers:
            raise ValueError("cascade needs at least one tier")
        self.tiers = list(tiers)
        self.budget = dict(budget_per_tier or {})
        self._used: dict[str, int] = {}

    def reset_budget(self) -> None:
        self._used = {}

    def _budget_ok(self, tier_name: str) -> bool:
        cap = self.budget.get(tier_name)
        if cap is None:
            return True
        return self._used.get(tier_name, 0) < cap

    def classify(self, edges: Sequence[EdgeContext]) -> CascadeReport:
        if not edges:
            return CascadeReport(results=[], telemetry=[])

        n = len(edges)
        accepted: list[TreatmentResult | None] = [None] * n
        fallbacks: list[TreatmentResult | None] = [None] * n
        tier_log: list[list[str]] = [[] for _ in range(n)]
        accepted_at: list[str] = [""] * n
        accepted_conf: list[float] = [0.0] * n
        latencies: list[float] = [0.0] * n

        pending: list[int] = list(range(n))

        for classifier, threshold in self.tiers:
            if not pending:
                break

            this_pending: list[int] = []
            this_inputs: list[EdgeContext] = []
            for idx in pending:
                if not self._budget_ok(classifier.name):
                    this_pending.append(idx)  # keep for next tier
                    continue
                this_pending.append(idx)
                this_inputs.append(edges[idx])

            if not this_inputs:
                # Budget exhausted for this tier on every pending edge.
                pending = [i for i in pending if not self._budget_ok(classifier.name)]
                continue

            t0 = time.perf_counter()
            results = classifier.classify_batch(this_inputs)
            tier_latency = (time.perf_counter() - t0) * 1000.0
            input_iter = iter(zip(this_inputs, results, strict=True))

            still_pending: list[int] = []
            for idx in this_pending:
                if not self._budget_ok(classifier.name):
                    still_pending.append(idx)
                    continue
                _, res = next(input_iter)
                self._used[classifier.name] = self._used.get(classifier.name, 0) + 1
                tier_log[idx].append(classifier.name)
                latencies[idx] += tier_latency / max(len(this_inputs), 1)

                if res is None:
                    still_pending.append(idx)
                    continue
                # Track the best fallback we've seen so far.
                if fallbacks[idx] is None or res.confidence > fallbacks[idx].confidence:
                    fallbacks[idx] = res

                if res.confidence >= threshold:
                    accepted[idx] = res
                    accepted_at[idx] = classifier.name
                    accepted_conf[idx] = res.confidence
                else:
                    still_pending.append(idx)

            pending = still_pending

        # Fill in unfinished slots with their best fallback (or a neutral default).
        final_results: list[TreatmentResult] = []
        telemetry: list[CascadeTelemetry] = []
        for i in range(n):
            res = (
                accepted[i]
                or fallbacks[i]
                or TreatmentResult(
                    label="neutral",
                    confidence=0.0,
                    source="cascade:no-tier",
                )
            )
            final_results.append(res)
            telemetry.append(
                CascadeTelemetry(
                    edge_index=i,
                    accepted_tier=accepted_at[i] or res.source or "fallback",
                    accepted_confidence=accepted_conf[i] or res.confidence,
                    fallback_used=accepted[i] is None,
                    tier_attempts=tier_log[i],
                    total_latency_ms=latencies[i],
                )
            )

        return CascadeReport(results=final_results, telemetry=telemetry)
