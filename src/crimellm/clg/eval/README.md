# `clg.eval`

Gold-set evaluation for the retrieval + synthesis stack. Scores retrieval (recall@k, citation accuracy) and graph-specific behaviour (good-law precision/recall, as-of correctness) against a hand-curated YAML gold set.

## Files

| File | Purpose |
|---|---|
| `schema.py` | `GoldQuestion(question, jurisdiction, task_type, expected_authorities, expected_good_law, expected_treating_case, as_of, ...)`, `GoldSet` with `filter_by_jurisdiction(codes)` / `filter_by_task_type(types)` / `jurisdictions()` helpers, `load_gold_set(yaml_path) → GoldSet` |
| `runner.py` | `run_eval(gold_set, backend, synth) → EvalReport` — runs every question through `clg.retrieval.query.run_query` and collects `Answer` + metrics |
| `metrics.py` | `recall_at_k`, `citation_accuracy`, `good_law_precision_recall`, `as_of_correct`; aggregate per-question + corpus-wide |
| `report.py` | `to_markdown(report)`, `to_json(report)` for human + machine consumption |

## When to use

- Before merging a change to `seed`, `expand`, `rerank`, or `synthesize` → run `clg eval` and compare against the previous report.
- When swapping embedder (`clg embed-rebuild --backend ...`) → re-run eval to catch regressions.
- When tuning the treatment cascade thresholds → eval the `good_law_precision_recall` metric specifically.

## How

```bash
# Full sweep
clg eval --gold-set data/eval/seed.yaml --backend voyage --synth anthropic --format md --out report.md

# Per-jurisdiction (DK firm workflow): only DK + cross-jurisdiction questions
clg eval --gold-set data/eval/seed.yaml --jurisdiction DK,XJ --synth anthropic

# Targeted regression: good-law gate across all jurisdictions
clg eval --gold-set data/eval/seed.yaml --task-type good_law --synth anthropic

# Combine
clg eval --gold-set data/eval/seed.yaml --jurisdiction DK --task-type good_law,no_fabrication
```

Seed gold set composition (Phase 10, version 2): 24 questions — US 2, UK 4, EU 5, DK 10, cross-jurisdiction (`XJ`, where `jurisdiction: null`) 3. Civil-law DK good-law questions use `departed_from` / `criticised` labels; common-law jurisdictions use the full `overruled` / `reversed` / `distinguished` set.

Gold set YAML example (see `data/eval/seed.yaml` for the full set):

```yaml
questions:
  - id: dk-single-fact-databeskyttelse-art-6
    question: "Hvad regulerer databeskyttelseslovens § 6 stk. 1?"
    task_type: single_fact
    jurisdiction: DK
    as_of: 2024-06-01
    expected_authorities:
      - dk/lbk/2018/502/section/§6/stk.1

  - id: eu-good-law-keck-vs-dassonville
    question: "Did the CJEU depart from Dassonville in Keck Mithouard?"
    task_type: good_law
    jurisdiction: EU
    expected_authorities:
      - ECLI:EU:C:1993:905
      - ECLI:EU:C:1974:82
    expected_good_law:
      ECLI:EU:C:1974:82: departed_from   # civil-law / CJEU label
    expected_treating_case: ECLI:EU:C:1993:905

  - id: xj-implements-databeskyttelsesloven-gdpr
    question: "Hvilken EU-forordning gennemføres af databeskyttelsesloven?"
    task_type: multi_hop
    jurisdiction: null                    # cross-jurisdiction → bucket "XJ"
    expected_authorities:
      - dk/lbk/2018/502                   # DK Instrument
      - eu/celex/32016R0679               # EU Instrument (target of IMPLEMENTS)
```

```python
from crimellm.clg.eval.schema import load_gold_set
from crimellm.clg.eval.runner import run_eval
from crimellm.clg.eval.report import to_markdown

gold = load_gold_set("data/eval/seed.yaml")
report = run_eval(gold, backend="voyage", synth="anthropic")
print(to_markdown(report))
```

Use the JSON output for CI gating — fail the build if recall@k drops below a threshold.
