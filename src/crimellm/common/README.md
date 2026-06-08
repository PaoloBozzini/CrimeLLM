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

### `language.py` — DA vs EN detection

Multi-signal binary language classifier. Used by the clg query parser (synthesis-language routing) and any future caller that needs a single `(lang, confidence)` answer without dragging in a multi-megabyte language-id library.

| Symbol | Use |
|---|---|
| `detect_language(text) → (lang, confidence)` | DA / EN binary, pure stdlib. `lang` is ISO 639-1, `confidence` is the normalised margin in `[0.0, 1.0]`. Below an internal `_DA_MIN_CONFIDENCE` threshold (`0.15`) the result always defaults to `"en"` — Claude handles EN > DA, so EN is the safer fallback for downstream synthesis |
| `DA_ONLY_CHARS`, `DA_STOPWORDS`, `EN_STOPWORDS`, `DA_BIGRAMS`, `EN_BIGRAMS`, `DA_SUFFIXES` | The four signal tables, exposed as `frozenset` / `tuple` so callers can audit / unit-test the inputs |

**Four signals, weighted sum:**

1. **DA-only diacritics** (`æ/ø/å`) — heaviest weight (4.0). One hit is decisive.
2. **Stopword frequency** — top-40 lists per language; weighted by hit ratio over total tokens. Lists are disjoint (asymmetric signal).
3. **Character bigrams** — DA-distinctive (`sk / ld / rk / rd / lv`) vs EN-distinctive (`th / wh / qu / wr / kn / gh / ph / ck`).
4. **DA word-ending suffixes** — Danish inflections English doesn't share (`-ende`, `-else`, `-heden`, `-erne`).

**Drop-in upgrade path:** swap in `langdetect` (CLD2 port) or `langid.py` behind a shim that preserves the `(lang, confidence)` return contract — no caller changes needed.

Pure stdlib. No new dependencies.

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
- Routing synthesis or UI output by language → call `detect_language(text)` and gate on the returned `(lang, confidence)`.

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

```python
from crimellm.common import detect_language

lang, confidence = detect_language("Højesteret har afsagt en afgørelse.")
# → ("da", 0.95). EN fallback when undetermined; safe default for synthesis.
```
