"""Classifier pipeline — fine-tune + zero-shot + FAISS RAG.

The original CrimeLLM stack: a 3-class crime classifier (`no` / `yes` /
`unclear`), zero-shot LLM baselines, and a FAISS-backed legal retriever.

Installable via the ``[classifier]`` extra. The new ``crimellm.clg.*``
subpackage holds the graph-RAG pipeline; the two coexist and share
``crimellm.common`` + ``crimellm.env``.
"""

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
from .device import resolve_device, training_kwargs_for_device
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
