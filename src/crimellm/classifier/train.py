from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import evaluate
import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import Config
from .device import resolve_device, training_kwargs_for_device


@dataclass
class TrainResult:
    trainer: Trainer
    tokenizer: Any
    model: Any
    eval_metrics: dict


def _build_metrics():
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    return compute_metrics


def train(splits: DatasetDict, cfg: Config | None = None) -> TrainResult:
    cfg = cfg or Config()
    device_info = resolve_device()
    print(f"[crimellm] device: {device_info}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_len)

    train_ds = splits["train"].map(tokenize, batched=True)
    eval_ds = splits["test"].map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
    )

    if cfg.freeze_encoder:
        for p in model.base_model.parameters():
            p.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode = "head-only (linear probe)" if cfg.freeze_encoder else "full fine-tune"
    print(f"[crimellm] mode: {mode} | trainable {trainable:,} / {total:,}")

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
        seed=cfg.seed,
        **training_kwargs_for_device(device_info),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_build_metrics(),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("[crimellm] final eval:", metrics)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    return TrainResult(trainer=trainer, tokenizer=tokenizer, model=model, eval_metrics=metrics)
