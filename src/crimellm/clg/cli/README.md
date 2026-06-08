# `clg.cli`

Typer-based command dispatcher. Installed as the console script `clg` (entry point in `pyproject.toml`: `clg = "crimellm.clg.cli:app"`).

## Purpose

Single front door to every stage of the pipeline. Each phase (graph admin, ingest, parse, load, link, embed, query, eval) is one subcommand group so users do not need to import Python modules to run the pipeline.

## Files

| File | Commands |
|---|---|
| `__init__.py` | Top-level app + registers groups; defines `clg embed`, `clg embed-rebuild`, `clg query` (supports `--jurisdiction US\|EW\|UK\|EU\|DK` + `--lang en\|da` overrides; JSON mode exposes resolved jurisdiction / language / as_of for audit), `clg eval` |
| `graph.py` | `clg graph init / status / wipe / drop-schema / rebuild-vector-index / cites / cited-by / counts / search / provision-as-of` |
| `ingest.py` | `clg ingest courtlistener / courtlistener-status / courtlistener-index / legislation-uk / eurlex / retsinformation / domstol / karnov / find-case-law` |
| `parse.py` | `clg parse retsinformation / eurlex / domstol` (sanity check; `clg load` runs parse internally) |
| `load.py` | `clg load courtlistener / legislation-uk / eurlex / retsinformation / domstol` |
| `link.py` | `clg link citations / distill / train-distilled / treatment` — `treatment` auto-builds one rule classifier per enabled jurisdiction so DK + EU + common-law rule sets coexist without double-labelling |
| `_common.py` | Shared helpers — CSV jurisdiction parser, store factory, output formatting |

## When to use

- Anything user-facing: prefer the CLI over importing the Python modules. The CLI wires up `Settings`, `Neo4jStore`, and logging consistently.
- For programmatic use inside notebooks / tests, import the underlying functions directly (`graph.schema.apply_schema`, `ingest.eurlex.download`, `retrieval.query.run_query`, etc.).

## How

```bash
clg --help                          # top-level groups
clg graph --help                    # subcommand help
clg query "Is malice required for murder?" --jurisdiction US --top-k 6 --json
```

See [`../README.md`](../README.md) for the end-to-end pipeline command sequence.
