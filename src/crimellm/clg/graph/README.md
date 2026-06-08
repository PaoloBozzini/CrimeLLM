# `clg.graph`

Thin Neo4j backend. Isolates every Cypher write behind a small surface so the store could be swapped (e.g. for Memgraph) without touching ingest / parse / link / retrieval.

## Files

| File | Purpose |
|---|---|
| `driver.py` | `Neo4jStore` wrapping `neo4j.Driver`; `get_store()` singleton; `.connect()`, `.verify()`, `.session()`, `.run(cypher, **params)` |
| `schema.py` | `apply_schema()` — idempotent constraints (8), indexes (6), vector index on `Chunk.embedding`, jurisdiction seed nodes. **Seeds only jurisdictions present in `Settings.enabled_jurisdictions`**; never deletes disabled-jurisdiction nodes (operator opt-in required). `rebuild_vector_index(dim, drop_chunks=bool)` for embedder swaps. `drop_schema()` removes constraints/indexes (data untouched) |
| `loaders.py` | Batched `UNWIND MERGE` writers: `load_courts`, `load_cases`, `load_citations`, `load_chunks`, `load_instruments`, `load_provisions`, `load_interprets` (Case→Provision), `load_implements` (Instrument→Instrument). All idempotent; flatten provenance into entity nodes. `search_chunks` accepts `enabled_jurisdictions` to silence disabled jurisdictions at the retrieval boundary while preserving data |

## When to use

| Need | Function |
|---|---|
| Fresh database | `clg graph init` → `apply_schema()` |
| Change embedding dim | `clg graph rebuild-vector-index --dim N --drop-chunks --yes` |
| Verify connection | `Neo4jStore.verify()` |
| Bulk write parsed entities | `load_cases`, `load_provisions`, `load_citations`, ... |
| Custom Cypher | `store.run(cypher, **params)` |

## How

```python
from crimellm.clg.graph.driver import get_store
from crimellm.clg.graph.schema import apply_schema, rebuild_vector_index
from crimellm.clg.graph.loaders import load_cases, load_citations

store = get_store()
apply_schema(store)
load_cases(store, [case1, case2, ...])           # entities from clg.models
load_citations(store, [cit1, cit2, ...])
rebuild_vector_index(store, dim=1024, drop_chunks=True)
```

All writers accept lists of `clg.models` instances and call `.to_neo4j_props()` internally.

## Schema (current)

Nodes: `Jurisdiction`, `Court`, `Case`, `Instrument`, `Provision`, `Chunk`, `Concept`, `Judge`.
Edges: `DECIDED`, `CITES [treatment, citing_sentence, weight]`, `IN_JURISDICTION`, `PART_OF`, `INTERPRETS`, `IMPLEMENTS`, `MENTIONS`.
Indexes: unique id per node type; `Case(jurisdiction, decision_date)`; `Provision(valid_from)`, `Provision(section_path)`, composite `(instrument_id, section_path, valid_from)`; vector index on `Chunk.embedding` (cosine, dim from `Settings.embedding_dim`).
