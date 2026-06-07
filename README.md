# CrimeLLM

Two complementary pipelines under one `uv` package.

### Pipeline 1 — **classifier** (original)

Three-class crime classifier (`no` / `yes` / `unclear`) for short text snippets. Three independent paths:

1. **Fine-tune** — `law-ai/InLegalBERT` (or any HF encoder) + a 3-class head.
2. **Zero-shot LLM** — uniform `classify(text)` API over three backends:
   - `OllamaClassifier` (local, JSON-schema constrained decoding)
   - `AnthropicClassifier` (Claude API, forced tool-use for schema enforcement, prompt caching)
   - `AirLLMClassifier` (layer-by-layer disk offload; MLX on Mac, CUDA + bitsandbytes on NVIDIA)
3. **FAISS RAG** — `LegalRetriever` (FAISS + BGE-small) over a JSONL corpus built from:
   - **US Code** (Title 18 — criminal code) via govinfo.gov
   - **UK Acts of Parliament** (Fraud Act, Theft Act, Bribery Act, …) via legislation.gov.uk
   - **CourtListener** judgments (federal/state case law) via REST v4

Lives at the package root (`train.py`, `inference.py`, `rag.py`, `zero_shot.py`, `embed_probe.py`, `corpora.py`). Plus a frozen-embedding linear probe for quick base-model bake-offs.

### Pipeline 2 — **clg (Common Legal Graph)**

Neo4j graph RAG over US + UK primary law. Lives under `src/crimellm/clg/`. Encodes the **citation and treatment** network plus **point-in-time legislation**, so it can answer multi-hop, "is this still good law?", and "as of date X" questions that flat vector RAG cannot. See [`docs/phases.local.md`](docs/phases.local.md) for the phase tracker (local-only, untracked) and the section further down for `clg` usage.

### Why two ingest layers?

Both pipelines pull from CourtListener / legislation.gov.uk / govinfo, but they ingest at different granularities:

- `crimellm.corpora` (REST → JSONL) for the FAISS pipeline — convenient single-file snippet corpora for the classifier.
- `crimellm.clg.ingest.*` (bulk CSV / XML → Neo4j) for the graph pipeline — full edges, full provenance, full point-in-time history.

They share `crimellm.common.http` (retry + UA + resumable download) so neither pays for the other.

## Setup

Install [uv](https://docs.astral.sh/uv/), then pick the extras you need:

```bash
# Lean default — just enough for shared utilities.
uv sync

# Pipeline 1 (classifier + FAISS RAG + zero-shot baselines).
uv sync --extra classifier
uv sync --extra airllm        # optional: AirLLM zero-shot backend
uv sync --extra airllm-mlx    # optional: Mac MLX backend for AirLLM
uv sync --extra airllm-cuda   # optional: NVIDIA + 4/8-bit (bitsandbytes)

# Pipeline 2 (clg — Neo4j graph RAG).
uv sync --extra clg

# Cross-cutting.
uv sync --extra anthropic     # Claude API (used by both pipelines)

# Everything (= classifier + clg + anthropic).
uv sync --extra all

# Contributor tooling.
uv sync --extra dev           # pytest + ruff
```

Cross-platform PyTorch: macOS gets MPS-ready PyPI wheel; Windows auto-routes to the CUDA 12.1 index via `[tool.uv.sources]`. Edit `pyproject.toml` if you need another CUDA version.

### Secrets (`.env`)

```bash
cp .env.example .env
# fill in any of:
#   COURTLISTENER_API_TOKEN=...   (free at courtlistener.com)
#   GOVINFO_API_KEY=...           (free at api.data.gov; "DEMO_KEY" works at 30/hr)
#   ANTHROPIC_API_KEY=...         (console.anthropic.com)
```

Downloaders and `AnthropicClassifier` auto-load `.env` via `crimellm.load_env()` — no code change needed.

## Quickstart

### Zero-shot, no training

```python
from crimellm import AnthropicClassifier
clf = AnthropicClassifier()
clf.classify("he broke into the shop at night and stole money from the till")
# ZeroShotResult(label='yes', confidence=1.0, reasoning='theft / burglary')
```

### Zero-shot with RAG context

```python
from crimellm import (
    LegalRetriever, AirLLMClassifier, UK_CRIMINAL_ACTS,
    fetch_us_code_sections, download_courtlistener, download_uk_legislation, load_jsonl,
)

# 1. Build a corpus (one-time)
fetch_us_code_sections("data/corpora/usc18", [
    "USCODE-2023-title18-partI-chap51-sec1111",   # Murder
    "USCODE-2023-title18-partI-chap103-sec2113",  # Bank robbery
    "USCODE-2023-title18-partI-chap63-sec1341",   # Mail fraud
])
download_uk_legislation("data/corpora/uk", statutes=UK_CRIMINAL_ACTS)
download_courtlistener("data/corpora/cl", max_docs=50)

# 2. Index it (FAISS + BGE-small)
records = (
    load_jsonl("data/corpora/usc18.jsonl")
    + load_jsonl("data/corpora/uk.jsonl")
    + load_jsonl("data/corpora/cl.jsonl")
)
LegalRetriever.build(records, "data/corpora/legal")

# 3. Classify with retrieved context
retriever = LegalRetriever.load("data/corpora/legal")
clf = AirLLMClassifier(model_id="Qwen/Qwen2.5-7B-Instruct", retriever=retriever)
clf.classify("she defrauded investors with false statements about finances")
# reasoning will cite Fraud Act 2006 s.2 / 18 U.S.C. § 1341 / similar judgments
```

### Fine-tune

```python
from crimellm import load_sample_dataset, train, Config, Classifier
res = train(load_sample_dataset(), Config())
Classifier(res.tokenizer, res.model).classify("he stole the car")
```

### CLI

```bash
# Ingest corpora
python -m crimellm.corpora us-code       --out data/corpora/usc18
python -m crimellm.corpora uk            --out data/corpora/uk
python -m crimellm.corpora courtlistener --out data/corpora/cl --max-docs 100
```

### Notebooks

```bash
uv run python -m ipykernel install --user --name crimellm --display-name "CrimeLLM (uv)"
uv run jupyter lab notebooks/
```

Layout:

- `notebooks/classifier/` — original classifier + FAISS RAG demos:
  - `finetune.ipynb` — train + evaluate the InLegalBERT classifier
  - `base_model_bakeoff.ipynb` — macro-F1 across candidate base models
  - `embedding_probe.ipynb` — frozen-embedding + logistic regression probe
  - `zero_shot_llm.ipynb` — Ollama / Anthropic / AirLLM comparison
  - `rag_demo.ipynb` — full pipeline: ingest USC + UK + CL → FAISS → classify with vs. without RAG
- `notebooks/clg/` — graph RAG notebooks (Phase 3+ will populate this).

## Package layout

```
src/crimellm/
├── config.py         hyperparams (Config dataclass)
├── device.py         CUDA / MPS / CPU auto-detect, per-backend training kwargs
├── env.py            .env loader (load_env, find_dotenv)
├── data.py           CSV + built-in sample loader
├── train.py          fine-tune entrypoint (transformers.Trainer)
├── inference.py      Classifier wrapper for fine-tuned model
├── embed_probe.py    sentence-transformers + sklearn linear probe
├── zero_shot.py      OllamaClassifier, AnthropicClassifier, AirLLMClassifier
├── rag.py            LegalRetriever (FAISS IndexFlatIP), RetrievalHit
└── corpora.py        download_us_code, fetch_us_code_sections,
                      download_courtlistener, download_uk_legislation,
                      download_bailii (stub), parse_us_code, load_jsonl
```

## Data format

Training CSV — two columns:

| column | type | meaning |
|---|---|---|
| `text` | str | the memory / sentence |
| `label` | int | 0 = no, 1 = yes, 2 = unclear |

See `data/sample.csv`.

Corpus JSONL — one record per line:

```json
{"id": str, "text": str, "source": "us_code"|"uk_legislation"|"courtlistener",
 "citation": str, "type": "statute"|"judgment", "metadata": {...}}
```

## Device handling

`crimellm.device.resolve_device()` picks **CUDA → MPS → CPU**. `training_kwargs_for_device()` injects per-backend `TrainingArguments` (bf16/fp16 on NVIDIA, fp32 on MPS, disables `pin_memory` off-CUDA). `AirLLMClassifier` routes to MLX on Darwin and to CUDA elsewhere.

## `clg` — Common Legal Graph (Neo4j RAG, in progress)

Graph-backed retrieval over US + UK primary law. The graph encodes the citation-and-treatment network plus point-in-time legislation, so the pipeline can answer multi-hop, "is this still good law?", and "as of date X" questions that flat vector RAG cannot. Lives in `src/crimellm/clg/` alongside the existing FAISS retriever.

### Setup

```bash
docker compose up -d neo4j        # Neo4j 5.x (community) on bolt://localhost:7687
make install                      # uv sync --extra clg --extra dev
cp .env.example .env              # fill in NEO4J_PASSWORD, VOYAGE_API_KEY, ANTHROPIC_API_KEY
uv run clg graph init             # constraints + vector index + jurisdiction seeds
uv run clg graph status
uv run pytest -q
```

Browse Neo4j at http://localhost:7474 (user `neo4j`, password from `.env`).

### CLI surface (Phase 0 — most subcommands are stubs)

```
clg graph init | status | wipe --yes | drop-schema --yes
clg ingest courtlistener | uscode | legislation-uk | find-case-law
clg parse  uslm | akoma-ntoso
clg link   citations | treatment
clg embed
clg query "..." --jurisdiction EW --as-of 2021-06-01
clg eval
```

### Source-data licences (read before bulk-fetching)

- **CourtListener / CAP / US Code / eCFR** — permissive; be polite on rate limits.
- **legislation.gov.uk** — Open Government Licence v3.0; no key required.
- **Find Case Law (TNA)** — reading is open under the Open Justice Licence, but **programmatic bulk extraction or enrichment requires applying for the (free) computational-analysis licence**. The `clg ingest find-case-law` downloader refuses to run until `TNA_COMPUTATIONAL_LICENCE_ACCEPTED=1` is set in `.env`. Rate limit: ~1,000 requests / 5 min.

## License

MIT — see `LICENSE`.

Source corpora retain their own licences: US Code is public domain (OLRC), UK legislation is Open Government Licence v3.0, CourtListener opinions are CC0 / public domain.
