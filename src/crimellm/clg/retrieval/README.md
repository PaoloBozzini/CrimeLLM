# `clg.retrieval`

Graph-RAG query runner. Vector hits are only the seed: real answers come from traversing CITES + INTERPRETS edges, filtering by "good law" and `as-of` date, reranking, and synthesizing a grounded answer with strict citations.

## Pipeline

```
parse_query → seed (vector) → expand (graph) → good_law → rerank → synthesize → Answer
```

## Files

| File | Stage | Purpose |
|---|---|---|
| `parse_query.py` | parse | `Query(question, jurisdiction_hint, as_of_date)` from CLI / API input |
| `seed.py` | seed | `seed_from_chunks(query)` — vector search against the `Chunk` index, returns `Candidate` seeds |
| `expand.py` | expand | `expand_seeds(seeds)` — traverse `CITES` (forward + backward), `INTERPRETS` (Provision↔Case), pick the temporally correct Provision version |
| `good_law.py` | filter | `check_good_law(case)` → `GoodLawFlag` (`GOOD` / `OVERRULED` / `REVERSED` / `UNKNOWN`); reads treatment edges produced by [`clg.link`](../link/README.md) |
| `rerank.py` | rerank | `rerank(candidates, query)` — combine embedding score, citation centrality, recency |
| `synthesize.py` | synthesize | `Synthesizer` ABC. Backends: `AnthropicSynthesizer`, `OllamaSynthesizer`, `AirLLMSynthesizer`, `FakeSynthesizer`. `extract_citations()` + `check_citations()` enforce that every cited id appears in the retrieved set |
| `query.py` | runner | `run_query(question, jurisdiction=None, as_of=None, top_k=6) → Answer` — wires every stage above |

`Answer`: `text`, `citations` (list of normalised ids), `caveats` (e.g. "*R v X* overruled by *R v Y* in 2018"), `used_entities`.

## When to use

| Need | Function / CLI |
|---|---|
| Ad hoc vector-only lookup | `clg graph search "..."` |
| Full graph-RAG answer (synthesized) | `clg query "..."` |
| Inspect intermediate stages | import `seed_from_chunks` / `expand_seeds` directly |
| Build a service / notebook on top | `from crimellm.clg.retrieval.query import run_query` |
| Swap synthesis backend | pass `--synth anthropic|ollama|airllm|fake` |

## How

```bash
clg query "Is malice aforethought required for common law murder?" --jurisdiction US
clg query "What does GDPR Art. 17 say as of 2021-06-01?" --jurisdiction EU --as-of 2021-06-01 --top-k 8 --json
```

```python
from crimellm.clg.retrieval.query import run_query

ans = run_query(
    "Is the Theft Act 1968 s.1 still good law in England?",
    jurisdiction="EW",
    as_of="2024-01-01",
    top_k=6,
)
print(ans.text)
for cit in ans.citations: print(cit)
for c   in ans.caveats:   print(c)
```

Good-law gating is what separates this from flat FAISS retrieval: even if a 1970s case has the highest vector similarity, `check_good_law` will surface that it was overruled and the synthesizer will flag it in `caveats`.
