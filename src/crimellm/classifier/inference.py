from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .device import resolve_device


class Classifier:
    """Lightweight inference wrapper around a saved checkpoint."""

    def __init__(self, model_dir: str | Path, max_len: int = 256):
        device = resolve_device().device
        self.max_len = max_len
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, text: str) -> str:
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_len, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits
        return self.id2label[int(logits.argmax(-1))]

    @torch.no_grad()
    def predict_proba(self, text: str) -> dict[str, float]:
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_len, return_tensors="pt"
        ).to(self.device)
        probs = torch.softmax(self.model(**inputs).logits, dim=-1).squeeze(0).tolist()
        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}
