"""Phase 5.2 — distillation sampling + teacher labelling + CSV output.

No live model calls. We exercise the pipeline with a deterministic
``FixedTeacher`` and assert that the output CSV is shaped exactly the way
``classifier.data.load_dataset_from_csv`` expects (int label column,
``label_str`` matches `LABEL_TO_ID` etc.).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from crimellm.clg.link.distill import (
    DistillSample,
    label_distribution,
    label_with_teacher,
    write_training_csv,
)
from crimellm.clg.link.train_distilled import teacher_agreement
from crimellm.clg.link.treatment_base import (
    LABEL_TO_ID,
    TREATMENT_VOCAB,
    EdgeContext,
    TreatmentClassifier,
    TreatmentResult,
)


class _FixedTeacher(TreatmentClassifier):
    """Returns scripted labels; the missing slots = abstain (None)."""

    def __init__(self, name: str, scripted: list[TreatmentResult | None]):
        self.name = name
        self._scripted = scripted

    def classify_batch(self, edges):  # noqa: ANN001
        return list(self._scripted[: len(edges)])


def _edge(idx: int, sentence: str = "x") -> EdgeContext:
    return EdgeContext(
        citing_case_id=f"cl-citing-{idx}",
        cited_case_id=f"cl-cited-{idx}",
        cited_case_name=f"Cited Case {idx}",
        citing_sentence=sentence,
    )


def test_label_with_teacher_skips_abstentions() -> None:
    teacher = _FixedTeacher(
        "fake",
        [
            TreatmentResult(label="overruled", confidence=0.95, source="fake"),
            None,  # abstain → dropped from output
            TreatmentResult(label="followed", confidence=0.8, source="fake"),
        ],
    )
    edges = [_edge(0), _edge(1), _edge(2)]
    samples = label_with_teacher(edges, teacher=teacher)
    assert len(samples) == 2
    assert [s.label for s in samples] == ["overruled", "followed"]


def test_distill_sample_to_row_format() -> None:
    s = DistillSample(
        edge=_edge(0, sentence="The judgment is reversed."),
        label="reversed",
        confidence=0.92,
        teacher="fake",
    )
    row = s.to_row()
    # The "label" column has to be an int — classifier/data.py casts via ClassLabel.
    assert isinstance(row["label"], int)
    assert row["label"] == LABEL_TO_ID["reversed"]
    assert row["label_str"] == "reversed"
    assert "Cited:" in row["text"]
    assert "Sentence:" in row["text"]
    assert row["citing_case_id"] == "cl-citing-0"


def test_write_training_csv_round_trips(tmp_path: Path) -> None:
    samples = [
        DistillSample(
            edge=_edge(i, sentence=f"sentence {i}"),
            label=label,
            confidence=0.9,
            teacher="fake",
        )
        for i, label in enumerate(["overruled", "followed", "applied"])
    ]
    out = tmp_path / "treatment.csv"
    n = write_training_csv(samples, out)
    assert n == 3

    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 3
    assert {"text", "label", "label_str", "confidence", "teacher"}.issubset(rows[0].keys())
    # All labels are valid ints mapping back into the vocabulary.
    assert all(int(r["label"]) < len(TREATMENT_VOCAB) for r in rows)
    assert all(r["label_str"] in TREATMENT_VOCAB for r in rows)


def test_label_distribution_counts_correctly() -> None:
    samples = [
        DistillSample(edge=_edge(0), label="overruled", confidence=0.9, teacher="fake"),
        DistillSample(edge=_edge(1), label="overruled", confidence=0.8, teacher="fake"),
        DistillSample(edge=_edge(2), label="followed", confidence=0.7, teacher="fake"),
    ]
    assert label_distribution(samples) == {"overruled": 2, "followed": 1}


def test_teacher_agreement() -> None:
    assert teacher_agreement([0, 1, 2, 3], [0, 1, 5, 3]) == 0.75
    assert teacher_agreement([], []) == 0.0


def test_csv_is_consumable_by_classifier_loader(tmp_path: Path) -> None:
    """End-to-end: write a 12-row CSV (2-3 examples per class), feed into
    ``crimellm.classifier.load_dataset_from_csv`` and check shape.

    This is the contract between Phase 5.2's CSV writer and the existing
    classifier-stack loader — if it ever drifts, this test catches it.
    """
    try:
        from crimellm.classifier import load_dataset_from_csv  # noqa: F401
    except ImportError:
        pytest.skip("classifier extra not installed")

    # Build a balanced enough sample to keep train_test_split's stratify happy.
    samples = []
    for label in ("overruled", "followed"):
        for i in range(6):
            samples.append(
                DistillSample(
                    edge=_edge(i, sentence=f"{label} sentence variant {i}"),
                    label=label,
                    confidence=0.9,
                    teacher="fake",
                )
            )

    out = tmp_path / "treatment.csv"
    write_training_csv(samples, out)

    splits = load_dataset_from_csv(
        out,
        num_labels=len(TREATMENT_VOCAB),
        test_size=0.25,
        seed=42,
    )
    assert "train" in splits and "test" in splits
    assert len(splits["train"]) > 0
    assert len(splits["test"]) > 0
    # Labels should be integers in [0, num_labels).
    assert all(
        0 <= int(splits["train"][i]["label"]) < len(TREATMENT_VOCAB)
        for i in range(len(splits["train"]))
    )
