"""crimellm — two pipelines under one roof.

1. **classifier** (original) — fine-tune InLegalBERT / DeBERTa into a
   3-class crime classifier, zero-shot baselines (Ollama / Anthropic /
   AirLLM), and a FAISS-backed RAG retriever. Lives in
   ``crimellm.classifier``. Requires the ``[classifier]`` extra.

2. **clg — Common Legal Graph** (new, in progress) — Neo4j graph RAG over US
   + UK primary law with citation/treatment edges and point-in-time
   legislation. Lives in ``crimellm.clg``. Requires the ``[clg]`` extra.

Both stacks share ``crimellm.common.*`` (HTTP retry, streaming download) and
the ``.env`` loader (``crimellm.env``).

Public top-level re-exports keep existing notebooks working. Imports are
wrapped in try/except so a lean ``pip install crimellm[clg]`` (without
``[classifier]``) still imports cleanly — unavailable symbols simply aren't
re-exported.
"""

from __future__ import annotations

__version__ = "0.2.0"

# Always available (stdlib + dotenv).
from .env import find_dotenv, load_env

__all__: list[str] = ["__version__", "load_env", "find_dotenv"]

# --- classifier pipeline re-exports (best-effort) ---------------------------

try:
    from .classifier import (
        SYSTEM_PROMPT,
        UK_CRIMINAL_ACTS,
        AirLLMClassifier,
        AnthropicClassifier,
        Classifier,
        Config,
        LegalRetriever,
        OllamaClassifier,
        ProbeResult,
        RetrievalHit,
        ZeroShotResult,
        build_output_schema,
        download_bailii,
        download_courtlistener,
        download_uk_legislation,
        download_us_code,
        encode_texts,
        fetch_us_code_sections,
        linear_probe,
        load_dataset_from_csv,
        load_jsonl,
        load_sample_dataset,
        parse_us_code,
        resolve_device,
        train,
        training_kwargs_for_device,
    )

    __all__ += [
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
except ImportError:
    pass
