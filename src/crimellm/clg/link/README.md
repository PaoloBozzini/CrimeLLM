# `clg.link`

Two related jobs:

1. **Citation extraction** — find legal citations in free text and normalise them to graph ids.
2. **Treatment classification** — for each `CITES` edge, label *how* the citing case treats the cited case (followed, distinguished, overruled, ...).

## Files

### Citation extraction

| File | Purpose |
|---|---|
| `cite_registry.py` | Registry pattern. `extract_all(text, parsers=[...])` yields `CitationHit(raw, normalised_id, kind, span, jurisdiction)` |
| `cite_us.py` | US parser (eyecite-backed); registers on import |
| `cite_eu.py` | EU citation parser (CELEX, ECLI); registers on import |
| `cite_dk.py` | Danish citation parser (UfR, ELI); registers on import |
| `citation_context.py` | Extract the citing sentence around a citation hit — used as input for treatment classification and as training context for the distilled model |

### Treatment classification (cascade)

Treatment vocab (10 labels): common-law `followed, applied, considered, distinguished, doubted, not_followed, overruled, reversed, affirmed, neutral` + civil-law `departed_from, criticised`.

| File | Tier | Cost | Coverage | When |
|---|---|---|---|---|
| `treatment_base.py` | — | — | — | `TreatmentClassifier` ABC + `EdgeContext` dataclass |
| `treatment_rules.py` | 1 | free | ~35% | Regex / signal-phrase rules. Always tried first |
| `treatment_distilled.py` | 2 | cheap | high after training | InLegalBERT head fine-tuned on teacher labels. Needs `clg link distill` + `clg link train-distilled` first |
| `treatment_local_llm.py` | 3 | free | high | Ollama (any model) for cases the distilled head is unsure about |
| `treatment_anthropic.py` | 4 / teacher | paid | high | Claude Haiku with prompt caching. Used as teacher in distillation **and** as final tier |
| `treatment_cascade.py` | — | — | — | `CascadeClassifier` orchestrates the tiers with per-tier confidence thresholds; budgets Claude calls |

## When to use what

| Situation | Path |
|---|---|
| Extract citations from a body of text | `cite_registry.extract_all(text, parsers=[...])` |
| Label a single edge offline (fast) | `treatment_rules.RulesClassifier` |
| Cheap + accurate at scale | Distill once (`clg link distill` + `clg link train-distilled`), then run `treatment_distilled` |
| Production batch labelling | `treatment_cascade.CascadeClassifier` with `rules → distilled → ollama → anthropic` |
| Single ad hoc edge, no setup | `treatment_anthropic.AnthropicClassifier` |

## How

```bash
# 1. extract citations from text → CitationHit objects, written to graph as CITES edges
clg link citations --file case_body.txt --jurisdiction UK

# 2. sample edges and label them with the Claude teacher
clg link distill --sample 5000 --teacher anthropic --out data/training/treatment.csv

# 3. fine-tune the distilled head on those teacher labels
clg link train-distilled \
    --in data/training/treatment.csv \
    --base-model law-ai/InLegalBERT \
    --out artifacts/treatment_head

# 4. classify every unclassified edge in the graph via the cascade
clg link treatment \
    --backend rules+distilled+ollama \
    --confidence-threshold 0.85 \
    --distilled-dir artifacts/treatment_head
```

Treatment labels land on the `CITES` edge as the `treatment` property; the citing sentence is stored as `citing_sentence`. Downstream `clg.retrieval.good_law` reads `treatment` to gate "still good law?" answers.
