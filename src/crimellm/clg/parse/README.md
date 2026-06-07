# `clg.parse`

Convert raw source files (downloaded by [`clg.ingest`](../ingest/README.md)) into the typed document model in [`clg.models`](../models.py). Pure functions; no Neo4j writes — that happens in `clg load`.

## Files

| File | Input | Output | Notes |
|---|---|---|---|
| `courtlistener.py` | bz2 CSV streams (courts, clusters, opinions, citations.csv) | `Court`, `Case`, `Citation` tuples | Streams; builds opinion→cluster sidecar index on first run (slow, one-time per dump) |
| `legislation_uk.py` | CLML XML whole-act files | one `Instrument` per Act + one `Provision` per section per version | Sets `valid_from` / `valid_to` for point-in-time queries |
| `eurlex.py` | Akoma Ntoso / FORMEX XML | `Instrument` + `Provision` (regulations/directives) or `Case` (CJEU judgments) | Multi-language; CELEX-keyed |
| `retsinformation.py` | Danish primary-law XML | `Instrument` + `Provision` | Optionally explodes substykker into separate Provisions or folds into parent § text. Preamble EU references → CELEX → IMPLEMENTS seeds |
| `domstol.py` | DK judgment PDF or extracted text | `Case` + `citation_hits` | Two-layer: `parse_judgment_text` (pure-text) + `parse_judgment_pdf` (pypdf wrapper). Auto-recovers ECLI / court id / decision date from body via DA-aware regexes. Feeds Phase 1 DK + EU citation parsers |
| `find_case_law.py` | TNA Akoma Ntoso XML | `Case` | Minimal; expanded in later phase |

## When to use

- During development / debugging: `clg parse <source> --file path/to/raw.xml ...` prints parsed entities so you can sanity-check before loading into Neo4j.
- Inside the pipeline: `clg load <source>` calls these parsers internally and streams entities into [`clg.graph.loaders`](../graph/README.md).
- Standalone: import a parser and iterate entities in a notebook.

## How

```bash
clg parse retsinformation --file data/raw/retsinformation/lbk-2018-502.xml \
    --doc-type lbk --year 2018 --num 502
clg parse eurlex          --file data/raw/eurlex/32016R0679_en.xml \
    --kind regulation --celex 32016R0679 --lang en
clg parse domstol         --file data/raw/domstol/ECLI_DK_HR_2023_1234.pdf
```

```python
from crimellm.clg.parse.legislation_uk import parse_act
for entity in parse_act("data/raw/legislation_uk/ukpga-2006-35.xml", version="current"):
    print(entity)            # Instrument or Provision dataclass
```

All emitted entities have `.to_neo4j_props()` and can be passed straight to [`clg.graph.loaders`](../graph/README.md).
