"""``crimellm.classifier.config.Config`` — dataclass round-trip + helpers.

Imports the config module directly so the test runs even when the heavy
``[classifier]`` extra (transformers / torch / sentence-transformers) is not
installed — the top-level ``crimellm.classifier`` package would otherwise
fail to import.
"""

from __future__ import annotations

from crimellm.classifier.config import Config


def test_defaults() -> None:
    c = Config()
    assert c.num_train_epochs > 0
    assert c.train_batch_size > 0
    assert c.output_dir == "./artifacts/checkpoints"
    assert c.freeze_encoder is True
    assert c.num_labels == 3


def test_label_round_trip() -> None:
    c = Config()
    assert c.id2label == {0: "no", 1: "yes", 2: "unclear"}
    assert c.label2id == {"no": 0, "yes": 1, "unclear": 2}
    for k, v in c.id2label.items():
        assert c.label2id[v] == k


def test_id2label_override() -> None:
    c = Config(id2label={0: "neg", 1: "pos"})
    assert c.num_labels == 2
    assert c.label2id == {"neg": 0, "pos": 1}


def test_deprecated_root_shim_emits_warning(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    import importlib
    import warnings

    import crimellm.config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(crimellm.config)

    assert any(
        issubclass(w.category, DeprecationWarning) and "crimellm.config" in str(w.message)
        for w in caught
    )
    assert crimellm.config.Config is Config


def test_deprecated_training_config_shim_emits_warning() -> None:
    import importlib
    import warnings

    import crimellm.training_config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(crimellm.training_config)

    assert any(
        issubclass(w.category, DeprecationWarning) and "crimellm.training_config" in str(w.message)
        for w in caught
    )
    assert crimellm.training_config.Config is Config
