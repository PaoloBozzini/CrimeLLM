# `clg.embed`

Chunk Provisions and Cases, embed the chunks, and write them to the vector index.

## Files

| File | Purpose |
|---|---|
| `chunker.py` | `chunk_provision(provision)` — semantic chunking of statute section text → `Chunk` nodes; `chunk_case(case)` for case bodies. Chunks inherit `parent_id` + `parent_type` |
| `embedder.py` | Swappable `Embedder` ABC. Backends: `VoyageEmbedder`, `OpenAIEmbedder`, `SentenceTransformerEmbedder` (local, GPU-friendly), `FakeEmbedder` (dev). `get_embedder(backend, model, device)` auto-selects: Voyage if `VOYAGE_API_KEY` set, else local Qwen / BGE, else fake. `embed_in_batches()` streams vectors |

## When to use

- After `clg load` populates `Provision` and `Case` nodes → run `clg embed` to materialise chunks + vectors.
- After changing `EMBEDDING_MODEL` or `EMBEDDING_DIM` → `clg graph rebuild-vector-index --dim N --drop-chunks --yes` then `clg embed-rebuild --jurisdiction ... --backend ...`.
- For per-jurisdiction iteration during development → `clg embed --jurisdiction DK --limit 500`.

## Backend choice

| Need | Backend | Model |
|---|---|---|
| Fastest setup, cheapest | `voyage` | `voyage-3` (set `VOYAGE_API_KEY`) |
| Local / offline / GPU | `sentence-transformers` | `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-8B`, ... |
| OpenAI compatibility | `openai` | `text-embedding-3-large` |
| Unit tests | `fake` | deterministic random |

The active dimension is held in `Settings.embedding_dim` (auto-derived from model). When you switch models, `clg graph rebuild-vector-index` must be run with the new dim.

## How

```bash
clg embed --backend voyage
clg embed --backend sentence-transformers --model BAAI/bge-m3 --jurisdiction US --limit 5000
clg embed-rebuild --jurisdiction DK,EU --backend sentence-transformers --yes
```

```python
from crimellm.clg.embed.embedder import get_embedder
from crimellm.clg.embed.chunker import chunk_provision

embedder = get_embedder(backend="sentence-transformers", model="BAAI/bge-m3")
chunks = list(chunk_provision(provision))
vectors = embedder.embed_in_batches([c.text for c in chunks], batch_size=64)
```

Chunks are linked to their parent via the `PART_OF` edge and indexed in the vector index defined by [`clg.graph.schema`](../graph/README.md). They feed the seed step of [`clg.retrieval`](../retrieval/README.md).
