# `crimellm.clg` — Common Legal Graph

Neo4j graph-RAG pipeline over **US + EW/UK + EU + DK** primary law. Encodes the citation-and-treatment network plus point-in-time legislation, so it answers multi-hop, "is this still good law?", and "as of date X" questions that flat vector RAG cannot.

The sibling [`crimellm.classifier`](../classifier/README.md) is the original FAISS-only pipeline; `clg` replaces the retriever with a graph store and adds temporal + treatment reasoning.

## Submodule map

| Submodule | Purpose |
|---|---|
| `cli/` | Typer console script `clg` — all user-facing commands |
| `config.py` | Pydantic settings (`Settings`) — loads `.env`, jurisdiction toggle, model defaults |
| `models.py` | Document model: `Jurisdiction`, `Case`, `Instrument`, `Provision`, `Chunk`, `Citation`, `Court`, `Treatment`, `Provenance`. All have `.to_neo4j_props()` |
| `graph/` | Neo4j backend — `driver.py` (`Neo4jStore`, `get_store()`), `schema.py` (constraints, indexes, vector index, jurisdiction seeds), `loaders.py` (batched `UNWIND MERGE`) |
| `ingest/` | Source downloaders — `_base.Source` ABC; one file per source: `courtlistener`, `legislation_uk`, `eurlex`, `retsinformation`, `find_case_law` |
| `parse/` | Source-format → `crimellm.clg.models` — `courtlistener` (bz2 CSV stream), `legislation_uk` (CLML XML), `eurlex` (Akoma Ntoso / FORMEX), `retsinformation`, `find_case_law` |
| `link/` | Citation extraction (`cite_registry`, `cite_us`, `cite_eu`, `cite_dk`, eyecite-backed) and treatment classification cascade (`treatment_rules` → `treatment_distilled` → `treatment_local_llm` → `treatment_anthropic`) |
| `embed/` | `chunker.py` (provision/case → `Chunk`), `embedder.py` (Voyage / OpenAI / SentenceTransformer / Fake backends) |
| `retrieval/` | Graph-RAG runner — `parse_query`, `seed` (vector hits), `expand` (CITES / INTERPRETS / temporal), `good_law` (overruled / reversed gating), `rerank`, `synthesize` (Anthropic / Ollama / AirLLM / Fake), `query.run_query` |
| `eval/` | Gold-set runner — `schema` (`GoldQuestion`, `GoldSet`), `runner`, `metrics` (recall@k, citation accuracy, good-law P/R, as-of correctness), `report` (md/json) |

## Jurisdictions

| Code | Source | Notes |
|---|---|---|
| `US` | CourtListener bulk dumps | Federal + state courts, citations CSV |
| `EW` / `UK` | legislation.gov.uk (CLML), Find Case Law (TNA) | UK Acts; point-in-time versions (`enacted`, `current`, ISO dates) |
| `EU` | EUR-Lex CELLAR (Akoma Ntoso / FORMEX) | CELEX-keyed, 24 languages |
| `DK` | retsinformation.dk | `lov`, `lbk`, `bek`; ELI-keyed |

Toggle via `ENABLED_JURISDICTIONS=US,EW,UK,EU,DK` in `.env`.

## Graph schema (sketch)

Nodes: `Jurisdiction`, `Court`, `Case`, `Instrument`, `Provision`, `Chunk`, `Concept`, `Judge`.
Edges: `DECIDED` (Case→Court), `CITES` (Case→Case, `[treatment, citing_sentence, weight]`), `IN_JURISDICTION`, `PART_OF` (Chunk→parent), `INTERPRETS` (Case→Provision), `IMPLEMENTS` (Case→Instrument), `MENTIONS` (Case→Concept).
Indexes: unique id per node type; case (jurisdiction, decision_date); provision (`valid_from`, `section_path`, composite); vector index on `Chunk.embedding` (cosine, dim from `EMBEDDING_DIM`).

## Setup

```bash
docker compose up -d neo4j         # bolt://localhost:7687, browser http://localhost:7474
uv sync --extra clg                # neo4j, typer, pydantic, eyecite, voyageai, anthropic, ...
cp .env.example .env               # fill in keys below
uv run clg graph init              # constraints + vector index + jurisdiction seeds
uv run clg graph status
```

### `.env`

```
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=crimellm-dev
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...                       # optional; else local sentence-transformers
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B     # or BAAI/bge-m3, etc.
ENABLED_JURISDICTIONS=US,EW,UK,EU,DK
TNA_COMPUTATIONAL_LICENCE_ACCEPTED=0        # set 1 only after applying for TNA licence
```

## End-to-end pipeline

```bash
# 0. bootstrap
clg graph init

# 1. ingest raw sources (each is resumable + rate-limited)
clg ingest courtlistener   --date 2024-12-31 --files courts,dockets,clusters,opinions,citations
clg ingest legislation-uk  --versions enacted,current --statutes ukpga/2006/35,ukpga/1968/60
clg ingest eurlex          --celex 32016R0679,32019L0770 --lang en,da
clg ingest retsinformation --items lbk/2018/502,lov/2023/1100
clg ingest find-case-law   # requires TNA_COMPUTATIONAL_LICENCE_ACCEPTED=1

# 2. parse (sanity check; load runs parse internally)
clg parse retsinformation --file data/raw/retsinformation/lbk-2018-502.xml --doc-type lbk --year 2018 --num 502
clg parse eurlex          --file data/raw/eurlex/32016R0679_en.xml --kind regulation --celex 32016R0679 --lang en

# 3. load into Neo4j
clg load courtlistener    --date 2024-12-31 --limit 1000
clg load legislation-uk   --versions enacted,current
clg load eurlex           --celex 32016R0679,32019L0770 --lang en,da
clg load retsinformation  --items lbk/2018/502 --explode-subparagraphs

# 4. citations + treatment edges
clg link citations  --file case_body.txt --jurisdiction UK
clg link distill    --sample 5000 --teacher anthropic --out data/training/treatment.csv
clg link train-distilled --in data/training/treatment.csv --base-model law-ai/InLegalBERT --out artifacts/treatment_head
clg link treatment  --backend rules+distilled+ollama --distilled-dir artifacts/treatment_head

# 5. embed chunks
clg embed         --backend voyage
clg embed-rebuild --jurisdiction DK,EU --backend sentence-transformers --yes

# 6. query
clg query "Is malice aforethought required for common law murder?" --jurisdiction US
clg query "What does GDPR Art. 17 say as of 2021-06-01?" --jurisdiction EU --as-of 2021-06-01 --json

# 7. eval
clg eval --gold-set data/eval/seed.yaml --backend voyage --synth anthropic --format md --out report.md
```

## When to use which command

| Need | Command |
|---|---|
| Fresh database / change schema dim | `clg graph init`, `clg graph rebuild-vector-index --dim <N> --drop-chunks --yes` |
| See what's loaded | `clg graph status`, `clg graph counts <case_id>` |
| Inspect citation neighbourhood | `clg graph cites <case_id>`, `clg graph cited-by <case_id>` |
| Inspect legislation at a date | `clg graph provision-as-of -i <instrument> -s <section> --as-of YYYY-MM-DD` |
| Vector-only ad hoc lookup | `clg graph search "query" -k 5` |
| Full graph-RAG answer (synthesized) | `clg query "..."` |
| Swap embedder | `clg embed-rebuild --backend <name> --jurisdiction ...` |
| Score retrieval against gold set | `clg eval --gold-set ...` |

## Source licences (read before bulk-fetching)

- **CourtListener / US Code / eCFR** — permissive; respect rate limits.
- **legislation.gov.uk** — Open Government Licence v3.0; no key.
- **EUR-Lex** — Commission re-use notice; no key.
- **retsinformation.dk** — Civilstyrelsen open licence; no key.
- **Find Case Law (TNA)** — programmatic bulk extraction requires the free **computational-analysis licence**. `clg ingest find-case-law` refuses to run until `TNA_COMPUTATIONAL_LICENCE_ACCEPTED=1`. Rate limit ~1000 req / 5 min.
