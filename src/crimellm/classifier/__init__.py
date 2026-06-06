"""Classifier pipeline — fine-tune + zero-shot + FAISS RAG.

The original CrimeLLM stack: a 3-class crime classifier (`no` / `yes` /
`unclear`), zero-shot LLM baselines, and a FAISS-backed legal retriever.

Installable via the ``[classifier]`` extra. The new ``crimellm.clg.*``
subpackage holds the graph-RAG pipeline; the two coexist and share
``crimellm.common`` + ``crimellm.env``.

Heavy attributes (torch / transformers / sentence-transformers) are loaded
lazily via PEP 562 ``__getattr__`` so importing submodules that don't need
the ML stack — e.g. ``crimellm.classifier.config`` — works in a lean
``[clg]``-only install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.device import resolve_device, training_kwargs_for_device
    from .config import Config
    from .corpora import (
        UK_CRIMINAL_ACTS,
        download_bailii,
        download_courtlistener,
        download_uk_legislation,
        download_us_code,
        fetch_us_code_sections,
        load_jsonl,
        parse_us_code,
    )
    from .data import load_dataset_from_csv, load_sample_dataset
    from .embed_probe import ProbeResult, encode_texts, linear_probe
    from .inference import Classifier
    from .rag import LegalRetriever, RetrievalHit
    from .train import train
    from .zero_shot import (
        SYSTEM_PROMPT,
        AirLLMClassifier,
        AnthropicClassifier,
        OllamaClassifier,
        ZeroShotResult,
        build_output_schema,
    )

__all__ = [
    "Config",
    "resolve_device",
    "training_kwargs_for_device",
    "load_dataset_from_csv",
    "load_sample_dataset",
    "train",
    "Classifier",
    "ProbeResult",
    "encode_texts",
    "linear_probe",
    "LegalRetriever",
    "RetrievalHit",
    "ZeroShotResult",
    "OllamaClassifier",
    "AnthropicClassifier",
    "AirLLMClassifier",
    "SYSTEM_PROMPT",
    "build_output_schema",
    "download_us_code",
    "fetch_us_code_sections",
    "download_courtlistener",
    "download_uk_legislation",
    "download_bailii",
    "UK_CRIMINAL_ACTS",
    "parse_us_code",
    "load_jsonl",
]

# Map exported name -> (submodule path relative to this package, attr name).
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "resolve_device": ("..common.device", "resolve_device"),
    "training_kwargs_for_device": ("..common.device", "training_kwargs_for_device"),
    "Config": (".config", "Config"),
    "load_dataset_from_csv": (".data", "load_dataset_from_csv"),
    "load_sample_dataset": (".data", "load_sample_dataset"),
    "ProbeResult": (".embed_probe", "ProbeResult"),
    "encode_texts": (".embed_probe", "encode_texts"),
    "linear_probe": (".embed_probe", "linear_probe"),
    "Classifier": (".inference", "Classifier"),
    "LegalRetriever": (".rag", "LegalRetriever"),
    "RetrievalHit": (".rag", "RetrievalHit"),
    "train": (".train", "train"),
    "ZeroShotResult": (".zero_shot", "ZeroShotResult"),
    "OllamaClassifier": (".zero_shot", "OllamaClassifier"),
    "AnthropicClassifier": (".zero_shot", "AnthropicClassifier"),
    "AirLLMClassifier": (".zero_shot", "AirLLMClassifier"),
    "SYSTEM_PROMPT": (".zero_shot", "SYSTEM_PROMPT"),
    "build_output_schema": (".zero_shot", "build_output_schema"),
    "download_us_code": (".corpora", "download_us_code"),
    "fetch_us_code_sections": (".corpora", "fetch_us_code_sections"),
    "download_courtlistener": (".corpora", "download_courtlistener"),
    "download_uk_legislation": (".corpora", "download_uk_legislation"),
    "download_bailii": (".corpora", "download_bailii"),
    "UK_CRIMINAL_ACTS": (".corpora", "UK_CRIMINAL_ACTS"),
    "parse_us_code": (".corpora", "parse_us_code"),
    "load_jsonl": (".corpora", "load_jsonl"),
}


def __getattr__(name: str):
    from importlib import import_module

    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = target
    module = import_module(module_path, package=__name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
