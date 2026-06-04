from .config import Config
from .env import load_env, find_dotenv
from .device import resolve_device, training_kwargs_for_device
from .data import load_dataset_from_csv, load_sample_dataset
from .train import train
from .inference import Classifier
from .embed_probe import ProbeResult, encode_texts, linear_probe
from .zero_shot import (
    AirLLMClassifier,
    AnthropicClassifier,
    OllamaClassifier,
    ZeroShotResult,
    SYSTEM_PROMPT,
    build_output_schema,
)
from .rag import LegalRetriever, RetrievalHit
from .corpora import (
    download_us_code,
    fetch_us_code_sections,
    download_courtlistener,
    download_uk_legislation,
    download_bailii,
    parse_us_code,
    load_jsonl,
    UK_CRIMINAL_ACTS,
)

__all__ = [
    "Config",
    "load_env",
    "find_dotenv",
    "resolve_device",
    "training_kwargs_for_device",
    "load_dataset_from_csv",
    "load_sample_dataset",
    "train",
    "Classifier",
    "ProbeResult",
    "encode_texts",
    "linear_probe",
    "ZeroShotResult",
    "OllamaClassifier",
    "AnthropicClassifier",
    "AirLLMClassifier",
    "SYSTEM_PROMPT",
    "build_output_schema",
    "LegalRetriever",
    "RetrievalHit",
    "download_us_code",
    "fetch_us_code_sections",
    "download_courtlistener",
    "download_uk_legislation",
    "download_bailii",
    "UK_CRIMINAL_ACTS",
    "parse_us_code",
    "load_jsonl",
]
