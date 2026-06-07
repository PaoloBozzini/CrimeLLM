# `crimellm.common`

Shared utilities used by both [`crimellm.classifier`](../classifier/README.md) (FAISS pipeline) and [`crimellm.clg`](../clg/README.md) (Neo4j graph-RAG). Deliberately small and dependency-light so the two pipelines do not pay for each other.

## Files

### `http.py` — resilient HTTP

Wraps `httpx` for polite, resumable downloads against rate-limited public legal-data APIs.

| Symbol | Use |
|---|---|
| `UA` | `dict` — `User-Agent` header; always include in outgoing requests |
| `get_with_retry(client, url, *, params=None, max_retries=4, timeout=60.0, follow_redirects=True) → httpx.Response` | GET with exponential backoff; honours `Retry-After` on 429 |
| `stream_download(url, dest, *, headers=None, chunk=1MB, resume=True, desc=None) → Path` | Range-based resumable streaming download to `dest` |
| `write_jsonl(records, path) → int` | Stream an iterable of dicts to a JSONL file; returns record count |

Always available — no torch / no neo4j dependencies.

### `device.py` — PyTorch device selection

| Symbol | Use |
|---|---|
| `DeviceInfo` | dataclass: `device`, `backend` (`"cuda"` / `"mps"` / `"cpu"`), `name`, `supports_fp16`, `supports_bf16` |
| `resolve_device() → DeviceInfo` | Picks CUDA → MPS → CPU |
| `training_kwargs_for_device(info=None) → dict` | HF `TrainingArguments` kwargs per backend (bf16/fp16 flags, `pin_memory` off for non-CUDA) |

Requires `torch` (installed via `--extra classifier`). `__init__.py` re-exports these only when `torch` is importable, so a lean `--extra clg`-only install still works.

`classifier/device.py` is a deprecated shim re-exporting from here; import from `crimellm.common.device` in new code.

## When to use

- Writing a new ingest source → use `get_with_retry` + `stream_download` + `UA`.
- Writing any code that touches `torch` / HF `Trainer` → call `resolve_device()` and pass `training_kwargs_for_device()` into `TrainingArguments`.
- Persisting fetched records → `write_jsonl`.

## Example

```python
import httpx
from crimellm.common.http import UA, get_with_retry, stream_download, write_jsonl

with httpx.Client(headers=UA, timeout=60.0) as client:
    resp = get_with_retry(client, "https://api.example.gov/cases", params={"page": 1})
    records = resp.json()["results"]

write_jsonl(records, "data/raw/cases.jsonl")
stream_download("https://bulk.example.gov/dump.csv.bz2", "data/raw/dump.csv.bz2")
```

```python
from crimellm.common.device import resolve_device, training_kwargs_for_device
from transformers import TrainingArguments

info = resolve_device()
args = TrainingArguments(output_dir="ckpt", **training_kwargs_for_device(info))
```
