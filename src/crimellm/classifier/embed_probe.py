"""Frozen-embedding + sklearn linear-probe utilities.

For models whose pretraining objective already gives meaningful sentence-level
vectors (sentence-transformers, BGE, E5, Qwen3-Embedding, ...), encoding once
and fitting a small classifier on top is usually faster, lighter on memory,
and competitive-or-better than fine-tuning a transformer head on small data.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .device import resolve_device


@dataclass
class ProbeResult:
    model_name: str
    macro_f1: float
    accuracy: float
    per_class_f1: list[float]
    embed_seconds: float
    fit_seconds: float
    report: str
    dim: int
    error: str | None = None


def _to_st_device(backend: str) -> str:
    # sentence-transformers accepts "cuda", "mps", "cpu"
    return backend if backend in {"cuda", "mps", "cpu"} else "cpu"


def encode_texts(
    model_name: str,
    texts: list[str],
    batch_size: int = 16,
    trust_remote_code: bool = False,
    normalize: bool = True,
) -> tuple[np.ndarray, SentenceTransformer]:
    """Encode a list of texts with a SentenceTransformer model."""
    info = resolve_device()
    st_kwargs = {"device": _to_st_device(info.backend)}
    if trust_remote_code:
        st_kwargs["trust_remote_code"] = True
    model = SentenceTransformer(model_name, **st_kwargs)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return vecs, model


def linear_probe(
    model_name: str,
    train_texts: list[str],
    train_labels: list[int],
    eval_texts: list[str],
    eval_labels: list[int],
    label_names: Iterable[str],
    batch_size: int = 16,
    trust_remote_code: bool = False,
    C: float = 1.0,
) -> ProbeResult:
    """Encode + fit LogisticRegression + report metrics. Catches errors per-model."""
    label_names = list(label_names)
    try:
        t0 = time.time()
        X_train, _ = encode_texts(
            model_name, train_texts, batch_size=batch_size, trust_remote_code=trust_remote_code
        )
        X_eval, _ = encode_texts(
            model_name, eval_texts, batch_size=batch_size, trust_remote_code=trust_remote_code
        )
        embed_seconds = time.time() - t0

        t1 = time.time()
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C, n_jobs=None).fit(
            X_train, train_labels
        )
        fit_seconds = time.time() - t1

        y_pred = clf.predict(X_eval)
        macro = float(f1_score(eval_labels, y_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(eval_labels, y_pred))
        per_class = [
            float(v)
            for v in f1_score(
                eval_labels,
                y_pred,
                labels=list(range(len(label_names))),
                average=None,
                zero_division=0,
            )
        ]
        report = classification_report(
            eval_labels, y_pred, target_names=label_names, zero_division=0
        )
        return ProbeResult(
            model_name=model_name,
            macro_f1=macro,
            accuracy=acc,
            per_class_f1=per_class,
            embed_seconds=embed_seconds,
            fit_seconds=fit_seconds,
            report=report,
            dim=int(X_train.shape[1]),
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            model_name=model_name,
            macro_f1=float("nan"),
            accuracy=float("nan"),
            per_class_f1=[],
            embed_seconds=float("nan"),
            fit_seconds=float("nan"),
            report="",
            dim=0,
            error=f"{type(e).__name__}: {e}",
        )
