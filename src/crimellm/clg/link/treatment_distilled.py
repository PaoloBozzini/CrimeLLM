"""Tier 2 — small encoder head fine-tuned on 10 treatment labels.

Reuses the classifier-stack fine-tune pipeline at ``num_labels=10`` with
the treatment vocab as id2label. Training happens in Phase 5.2
(``clg link train-distilled``); this module just *loads* the trained
model from disk and runs inference.

If the model directory doesn't exist yet, the classifier abstains on every
input (returns ``None``) — the cascade still works, it just doesn't get the
50%-coverage tier 2 boost.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from .treatment_base import (
    ID_TO_LABEL,
    TREATMENT_VOCAB,
    EdgeContext,
    TreatmentClassifier,
    TreatmentResult,
)


def _format_input(edge: EdgeContext) -> str:
    return (
        f"Cited: {edge.cited_case_name or edge.cited_case_id}\n"
        f"Sentence: {edge.citing_sentence or ''}"
    )


class DistilledTreatmentClassifier(TreatmentClassifier):
    """Tier 2: loads a HuggingFace seq-classifier head over the treatment vocab."""

    name = "distilled"

    def __init__(
        self,
        *,
        model_dir: str | Path,
        max_len: int = 256,
        device: str | None = None,
        batch_size: int = 32,
    ):
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as e:  # pragma: no cover — caller installs [classifier]
            raise ImportError(
                "transformers / torch not installed. Add the [classifier] extra "
                "before training or loading the distilled head."
            ) from e

        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"distilled treatment head not found at {path}. "
                "Run `clg link train-distilled --in data/training/treatment.csv` first."
            )

        self.model_dir = path
        self.max_len = max_len
        self.batch_size = batch_size

        if device is None:
            try:
                from ...common.device import resolve_device  # type: ignore

                device = resolve_device().backend
            except Exception:  # noqa: BLE001
                device = "cpu"
        self.device = device

        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self._model.eval()
        try:
            self._model.to(device)
        except Exception:  # noqa: BLE001 — bad device string -> stick to CPU
            self._model.to("cpu")
            self.device = "cpu"

    def classify_batch(self, edges: Sequence[EdgeContext]) -> list[TreatmentResult | None]:
        if not edges:
            return []
        import torch

        # The classifier abstains when sentence is empty (rules + distilled
        # don't add value over the LLM tiers in that case).
        slots: list[tuple[int, str]] = [
            (i, _format_input(e)) for i, e in enumerate(edges) if e.citing_sentence
        ]
        results: list[TreatmentResult | None] = [None] * len(edges)
        if not slots:
            return results

        indices = [i for i, _ in slots]
        texts = [t for _, t in slots]

        start = time.perf_counter()
        with torch.no_grad():
            for chunk_start in range(0, len(texts), self.batch_size):
                chunk = texts[chunk_start : chunk_start + self.batch_size]
                enc = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_ids = probs.max(dim=-1)
                for j in range(top_ids.shape[0]):
                    label_id = int(top_ids[j].item())
                    label = ID_TO_LABEL.get(label_id, "neutral")
                    if label not in TREATMENT_VOCAB:
                        label = "neutral"
                    conf = float(top_probs[j].item())
                    out_idx = indices[chunk_start + j]
                    results[out_idx] = TreatmentResult(
                        label=label,  # type: ignore[arg-type]
                        confidence=conf,
                        source=f"{self.name}:{self.model_dir.name}",
                        latency_ms=(time.perf_counter() - start) * 1000.0 / max(len(texts), 1),
                    )
        return results
