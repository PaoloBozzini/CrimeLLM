from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict


_SAMPLE = {
    "text": [
        "He took the neighbour's bike without asking and sold it.",
        "She paid for her groceries and walked home.",
        "They argued loudly in the street late at night.",
        "I forged my manager's signature on the expense form.",
        "We donated our old clothes to the shelter.",
        "Someone was in the room but it's not clear what they did.",
    ],
    "label": [1, 0, 2, 1, 0, 2],
}


def _cast_label(ds: Dataset, num_labels: int) -> Dataset:
    return ds.cast_column("label", ClassLabel(num_classes=num_labels))


def _split(ds: Dataset, test_size: float, seed: int) -> DatasetDict:
    split = ds.train_test_split(
        test_size=test_size, seed=seed, stratify_by_column="label"
    )
    return DatasetDict(train=split["train"], test=split["test"])


def load_sample_dataset(test_size: float = 0.33, seed: int = 42, num_labels: int = 3) -> DatasetDict:
    ds = _cast_label(Dataset.from_dict(_SAMPLE), num_labels)
    return _split(ds, test_size, seed)


def load_dataset_from_csv(
    path: str | Path,
    test_size: float = 0.33,
    seed: int = 42,
    num_labels: int = 3,
    text_col: str = "text",
    label_col: str = "label",
) -> DatasetDict:
    df = pd.read_csv(path)
    if text_col != "text" or label_col != "label":
        df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    ds = _cast_label(Dataset.from_pandas(df, preserve_index=False), num_labels)
    return _split(ds, test_size, seed)
