# CrimeLLM

Fine-tune an encoder (default: `law-ai/InLegalBERT`) into a 3-class classifier on the "is this memory a crime?" axis (`no` / `yes` / `unclear`). For a second axis (e.g. ethical good/bad), train a **separate** model with the same code by swapping the dataset — don't share a head.

## Layout

```
.
├── pyproject.toml          # uv project + cross-platform torch sources
├── .python-version         # 3.11
├── src/crimellm/           # importable package
│   ├── config.py           # hyperparams
│   ├── device.py           # CUDA / MPS / CPU auto-detect
│   ├── data.py             # CSV + built-in sample loader
│   ├── train.py            # train() entrypoint
│   └── inference.py        # Classifier wrapper
├── notebooks/finetune.ipynb
├── data/sample.csv
└── legacy/finetune_classifier.py   # original single-file script
```

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the project root:

### macOS (Apple Silicon — MPS)

```bash
uv sync
```

Default PyPI `torch` wheel ships MPS support. Auto-detected at runtime.

### Windows + NVIDIA (CUDA 12.1)

```powershell
uv sync
```

`pyproject.toml` already routes `torch` to the PyTorch CUDA 12.1 index **on `sys_platform == 'win32'`** via `[tool.uv.sources]`, so the right wheel is fetched automatically. If you need a different CUDA, edit the `[[tool.uv.index]]` URL (e.g. `cu118`, `cu124`) in `pyproject.toml`.

### Linux / CPU

Plain `uv sync` gives the CPU/CUDA-on-linux PyPI wheel.

## Run

### Jupyter notebook

```bash
uv run python -m ipykernel install --user --name crimellm --display-name "CrimeLLM (uv)"
uv run jupyter lab notebooks/finetune.ipynb
```

Pick the **CrimeLLM (uv)** kernel.

### Script

```bash
uv run python -c "from crimellm import load_sample_dataset, train, Config; train(load_sample_dataset(), Config())"
```

## Data format

CSV with two columns:

| column | type | meaning |
|--------|------|---------|
| text   | str  | the memory / sentence |
| label  | int  | 0 = no, 1 = yes, 2 = unclear |

See `data/sample.csv`.

## Device handling

`crimellm.device.resolve_device()` picks **CUDA → MPS → CPU**. `training_kwargs_for_device()` then injects per-backend `TrainingArguments` (bf16/fp16 on NVIDIA, fp32 on MPS, disables `pin_memory` off-CUDA).
