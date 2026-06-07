# `crimellm.classifier`

Three-class crime classifier (`no` / `yes` / `unclear`) for short text snippets, plus a FAISS-backed legal retriever used as RAG context. This is the original (v0.2) pipeline; the newer graph-based pipeline lives under [`crimellm.clg`](../clg/README.md).

## Purpose

Decide whether a piece of text describes a crime, in three independent ways:

1. **Supervised fine-tune** — `law-ai/InLegalBERT` (or any HF encoder) + a 3-class head.
2. **Zero-shot LLM** — uniform `classify(text)` API over three backends (Ollama, Anthropic, AirLLM).
3. **FAISS RAG** — dense retrieval over a JSONL corpus of US Code, UK Acts, and CourtListener opinions, injected as context into any zero-shot backend.

A frozen-embedding linear probe (`embed_probe.py`) gives a quick base-model bake-off without training a head.

## File map

| File | Purpose |
|---|---|
| `config.py` | `Config` dataclass — model name, lr, epochs, label map, `freeze_encoder` flag |
| `data.py` | `load_sample_dataset()` (in-memory demo), `load_dataset_from_csv()` (stratified split → HF `DatasetDict`) |
| `train.py` | `train(splits, cfg) → TrainResult` — HF `Trainer`, accuracy + macro-F1, optional encoder freeze |
| `inference.py` | `Classifier(model_dir)` — `.predict(text)`, `.predict_proba(text)` on a fine-tuned checkpoint |
| `embed_probe.py` | `encode_texts(...)`, `linear_probe(...) → ProbeResult` — sentence-transformers + sklearn logistic regression |
| `zero_shot.py` | `OllamaClassifier`, `AnthropicClassifier`, `AirLLMClassifier`, `ZeroShotResult`, `SYSTEM_PROMPT`, `build_output_schema` |
| `rag.py` | `LegalRetriever.build(...)`, `LegalRetriever.load(...)`, `.retrieve(query, k)` — FAISS `IndexFlatIP` + BGE-small |
| `corpora.py` | `download_us_code`, `fetch_us_code_sections`, `download_courtlistener`, `download_uk_legislation`, `parse_us_code`, `load_jsonl`, `UK_CRIMINAL_ACTS`; CLI `python -m crimellm.corpora ...` |
| `device.py` | Deprecated shim — re-exports `crimellm.common.device` |
| `__init__.py` | PEP 562 lazy exports (defers `torch`, `transformers`, `faiss` import until first access) |

## When to use which path

| Situation | Path |
|---|---|
| Have labelled CSV, want best in-domain accuracy | **Fine-tune** (`train.py`) |
| No labels, fast baseline on Mac/laptop | **Zero-shot Ollama** |
| No labels, cheap API, prompt caching | **Zero-shot Anthropic** |
| Want big open-weight model on small GPU | **Zero-shot AirLLM** (MLX on Mac, bitsandbytes on NVIDIA) |
| Need legal references in the prompt | Any zero-shot + `LegalRetriever` |
| Comparing candidate encoders before training | **Frozen probe** (`embed_probe.py`) |

## Setup

```bash
uv sync --extra classifier         # core: torch, transformers, datasets, faiss
uv sync --extra anthropic          # for AnthropicClassifier
uv sync --extra airllm             # for AirLLMClassifier (CPU)
uv sync --extra airllm-mlx         # Mac MLX backend
uv sync --extra airllm-cuda        # NVIDIA + 4/8-bit bitsandbytes
```

Optional `.env` keys (auto-loaded by `crimellm.env.load_env()`):
- `ANTHROPIC_API_KEY` — required for `AnthropicClassifier`
- `GOVINFO_API_KEY` — US Code; falls back to `DEMO_KEY` (30 req/hr)
- `COURTLISTENER_API_TOKEN` — enables full opinion bodies (snippet-only otherwise)

For Ollama: `brew install ollama && ollama pull qwen2.5:3b-instruct && ollama serve`.

## Usage

### Fine-tune

```python
from crimellm import Config, load_sample_dataset, train, Classifier

res = train(load_sample_dataset(), Config())          # default: full fine-tune InLegalBERT
clf = Classifier(res.tokenizer, res.model)
clf.classify("he stole the car")                       # → "yes"
```

CSV input — two columns `text` (str), `label` (int: 0=no, 1=yes, 2=unclear). See `data/sample.csv`.

### Zero-shot

```python
from crimellm import AnthropicClassifier, OllamaClassifier, AirLLMClassifier

clf = AnthropicClassifier()                            # claude-haiku-4-5 + prompt caching
clf.classify("broke into the shop at night and stole money")
# → ZeroShotResult(label='yes', confidence=1.0, reasoning='burglary/theft')

OllamaClassifier(model="qwen2.5:3b-instruct").classify("paid for groceries")
AirLLMClassifier(model_id="Qwen/Qwen2.5-7B-Instruct").classify("...")
```

All three backends return the same `ZeroShotResult(label, confidence, reasoning, raw, error)` and accept `retriever: LegalRetriever` to inject RAG context.

### FAISS RAG

```python
from crimellm import (
    LegalRetriever, fetch_us_code_sections, download_uk_legislation,
    download_courtlistener, load_jsonl, UK_CRIMINAL_ACTS, AnthropicClassifier,
)

# 1. build corpora (one-time)
fetch_us_code_sections("data/corpora/usc18", [
    "USCODE-2023-title18-partI-chap51-sec1111",
    "USCODE-2023-title18-partI-chap63-sec1341",
])
download_uk_legislation("data/corpora/uk", statutes=UK_CRIMINAL_ACTS)
download_courtlistener("data/corpora/cl", max_docs=50)

# 2. build FAISS index
records = (
    load_jsonl("data/corpora/usc18.jsonl")
    + load_jsonl("data/corpora/uk.jsonl")
    + load_jsonl("data/corpora/cl.jsonl")
)
LegalRetriever.build(records, "data/corpora/legal")

# 3. classify with retrieved context
retriever = LegalRetriever.load("data/corpora/legal")
clf = AnthropicClassifier(retriever=retriever)
clf.classify("she defrauded investors with false statements about finances")
# reasoning will cite Fraud Act 2006 s.2 / 18 U.S.C. § 1341
```

### Quick base-model bake-off

```python
from crimellm import linear_probe
probe = linear_probe(
    "BAAI/bge-small-en-v1.5",
    train_texts, train_labels, eval_texts, eval_labels,
)
print(probe.macro_f1, probe.per_class_f1)
```

### Corpus CLI

```bash
python -m crimellm.corpora us-code       --out data/corpora/usc18
python -m crimellm.corpora uk            --out data/corpora/uk
python -m crimellm.corpora courtlistener --out data/corpora/cl --max-docs 100
```

## Corpus JSONL schema

```json
{"id": "...", "text": "...", "source": "us_code|uk_legislation|courtlistener",
 "citation": "18 U.S.C. § 1111", "type": "statute|judgment", "metadata": {...}}
```

## Notes

- Lazy imports: `from crimellm import Config` does not pull in `torch`/`faiss` until the symbol is touched — safe in lean `clg`-only installs.
- Device handling: `resolve_device()` picks CUDA → MPS → CPU; `training_kwargs_for_device()` injects bf16/fp16 and `pin_memory` per backend.
- Schemas: all three zero-shot backends use JSON-schema constrained output (`build_output_schema`) so labels are guaranteed to be one of `no|yes|unclear`.
