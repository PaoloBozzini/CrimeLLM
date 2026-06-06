"""Phase 5.2 — train the distilled treatment-classification head.

Thin wrapper around ``crimellm.classifier.train.train`` at ``num_labels=10``
with the treatment vocabulary mapped onto ``Config.id2label``. The same
``train.py`` that fine-tunes the 3-class crime classifier handles the 10-class
treatment head — only the labels change.

Lazy-imports the classifier stack so a clg-only install still imports the
module; the heavy dep error only fires when the function is actually called.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from .treatment_base import ID_TO_LABEL, TREATMENT_VOCAB


@dataclass(slots=True)
class DistillTrainResult:
    output_dir: Path
    eval_metrics: dict[str, float]
    per_label_f1: dict[str, float]
    n_train: int
    n_test: int
    base_model: str


def _holdout_metrics(trainer: Any) -> tuple[dict[str, float], dict[str, float]]:
    """Compute macro-F1 + per-label F1 on the trainer's eval split."""
    preds = trainer.predict(trainer.eval_dataset)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    labels = list(range(len(TREATMENT_VOCAB)))
    per_label = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_label_dict = {ID_TO_LABEL[i]: float(v) for i, v in zip(labels, per_label, strict=True)}
    report_dict = {"macro_f1": macro}
    return report_dict, per_label_dict


def train_distilled_head(
    csv_path: str | Path,
    *,
    base_model: str = "law-ai/InLegalBERT",
    output_dir: str | Path = "artifacts/treatment_head",
    epochs: int = 4,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 256,
    test_size: float = 0.15,
    seed: int = 42,
    freeze_encoder: bool = False,
) -> DistillTrainResult:
    """Run the fine-tune, save under ``output_dir``, return held-out metrics.

    ``freeze_encoder=True`` trains only the classification head — much faster
    + less data-hungry but lower ceiling. Recommended when you only have a
    few hundred teacher labels; switch to full fine-tune (``False``) for 5k+.
    """
    try:
        from ...classifier import Config, load_dataset_from_csv, train
    except ImportError as e:  # pragma: no cover — caller installs [classifier]
        raise ImportError(
            "Distillation training needs the [classifier] extra "
            "(transformers / torch / datasets). Install with "
            "`uv sync --extra clg --extra classifier --extra dev`."
        ) from e

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"training CSV not found at {csv_path}")

    cfg = Config(
        model_name=base_model,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        max_len=max_len,
        output_dir=str(output_dir),
        test_size=test_size,
        seed=seed,
        freeze_encoder=freeze_encoder,
        id2label=dict(ID_TO_LABEL),
    )

    splits = load_dataset_from_csv(
        csv_path,
        test_size=test_size,
        seed=seed,
        num_labels=len(TREATMENT_VOCAB),
    )
    result = train(splits, cfg)
    report_dict, per_label_dict = _holdout_metrics(result.trainer)

    # Save the trainer's tokenizer + model under output_dir. classifier/train.py
    # already calls Trainer.save_model() + tokenizer.save_pretrained(); ensure
    # the path is present and reported.
    out_path = Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics = dict(result.eval_metrics or {})
    metrics.update(report_dict)
    return DistillTrainResult(
        output_dir=out_path,
        eval_metrics=metrics,
        per_label_f1=per_label_dict,
        n_train=len(splits["train"]),
        n_test=len(splits["test"]),
        base_model=base_model,
    )


def classification_report_text(per_label_f1: dict[str, float]) -> str:
    """Human-readable per-label F1 table for the CLI output."""
    rows = ["| label | F1 |", "|---|---|"]
    for label in TREATMENT_VOCAB:
        rows.append(f"| {label} | {per_label_f1.get(label, 0.0):.3f} |")
    return "\n".join(rows)


# Re-export for the CLI smoke import even when sklearn isn't installed.
def teacher_agreement(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Fraction of held-out edges where student matches teacher label.

    The brief Phase 5b held-out target is ≥ 95% agreement on edges that
    escalated past the rule tier.
    """
    yt = list(y_true)
    yp = list(y_pred)
    if not yt:
        return 0.0
    n_match = sum(1 for a, b in zip(yt, yp, strict=True) if a == b)
    return n_match / len(yt)
