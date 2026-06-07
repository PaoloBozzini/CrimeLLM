# `clg.eval`

Gold-set evaluation for the retrieval + synthesis stack. Scores retrieval (recall@k, citation accuracy) and graph-specific behaviour (good-law precision/recall, as-of correctness) against a hand-curated YAML gold set.

## Files

| File | Purpose |
|---|---|
| `schema.py` | `GoldQuestion(question, expected_case_ids, expected_provision_ids, as_of_date, ...)`, `GoldSet`, `load_gold_set(yaml_path) → GoldSet` |
| `runner.py` | `run_eval(gold_set, backend, synth) → EvalReport` — runs every question through `clg.retrieval.query.run_query` and collects `Answer` + metrics |
| `metrics.py` | `recall_at_k`, `citation_accuracy`, `good_law_precision_recall`, `as_of_correct`; aggregate per-question + corpus-wide |
| `report.py` | `to_markdown(report)`, `to_json(report)` for human + machine consumption |

## When to use

- Before merging a change to `seed`, `expand`, `rerank`, or `synthesize` → run `clg eval` and compare against the previous report.
- When swapping embedder (`clg embed-rebuild --backend ...`) → re-run eval to catch regressions.
- When tuning the treatment cascade thresholds → eval the `good_law_precision_recall` metric specifically.

## How

```bash
clg eval \
    --gold-set data/eval/seed.yaml \
    --backend voyage \
    --synth anthropic \
    --format md \
    --out report.md
```

Gold set YAML example:

```yaml
- question: "Is malice aforethought required for common-law murder in the US?"
  jurisdiction: US
  expected_case_ids: [scotus/1980/abc, ca9/2002/def]
  expected_provision_ids: [uscode/18/1111@2023-01-01]
  as_of: 2023-06-01

- question: "What does GDPR Art. 17 say?"
  jurisdiction: EU
  expected_provision_ids: [eurlex/32016R0679/art17@2018-05-25]
  as_of: 2021-06-01
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
