# `clg.ingest`

Source downloaders. One file per upstream system. Every downloader is resumable, rate-limited, cached on disk, and tags every record with provenance (source, source_url, source_id, retrieved_at).

## Files

| File | Source | Jurisdiction | Notes |
|---|---|---|---|
| `_base.py` | `Source` ABC, `IngestContext`, `LoadReport` | — | New sources inherit from `Source` and implement `download()` / `parse()` / `load()` |
| `courtlistener.py` | storage.courtlistener.com bulk CSVs (courts, dockets, clusters, opinions, citations) | `US` | Daily dumps keyed by `YYYY-MM-DD`; resumable bz2 streams |
| `legislation_uk.py` | legislation.gov.uk CLML XML | `EW` / `UK` | Whole-act XML; point-in-time versions (`enacted`, `current`, ISO dates) |
| `eurlex.py` | EUR-Lex / CELLAR | `EU` | Per-CELEX, per-language, per-format (Akoma Ntoso / FORMEX); no API key |
| `retsinformation.py` | retsinformation.dk | `DK` | `lov`, `lbk`, `bek`; keyed by ELI (`doc_type/year/num`); open licence |
| `domstol.py` | domstol.dk (Højesteret + Landsret) | `DK` | PDF-heavy; operator-supplied `(ECLI, URL)` list; seeds `DK_COURTS` hierarchy (HR / OLR / VLR / byret) |
| `karnov.py` | Karnov Online (commercial reporter) | `DK` | **Skeleton**: refuses to construct unless `KARNOV_API_KEY` is set; real ingester deferred until firm subscription confirmed |
| `find_case_law.py` | caselaw.nationalarchives.gov.uk (TNA) | `EW` / `UK` | High Court + superior judgments. **Gated**: refuses to run unless `TNA_COMPUTATIONAL_LICENCE_ACCEPTED=1`. ~1000 req / 5 min |

## When to use

- One-time bulk download → CLI: `clg ingest <source> ...`.
- Incremental / scripted → import the source class and call `.download(ctx)` in a notebook.
- Inspect what was already fetched → `clg ingest courtlistener-status --date YYYY-MM-DD`.
- Build the opinion→cluster sidecar index (slow, one-time per CL dump) → `clg ingest courtlistener-index --date YYYY-MM-DD`.

## How

```bash
clg ingest courtlistener   --date 2024-12-31 --files courts,dockets,clusters,opinions,citations
clg ingest legislation-uk  --versions enacted,current --statutes ukpga/2006/35,ukpga/1968/60
clg ingest eurlex          --celex 32016R0679,32019L0770 --lang en,da
clg ingest retsinformation --items lbk/2018/502,lov/2023/1100
clg ingest domstol         --items 'ECLI:DK:HR:2023:1|https://domstol.dk/.../1.pdf'
clg ingest karnov                                           # skeleton; needs KARNOV_API_KEY
clg ingest find-case-law                                    # set TNA_COMPUTATIONAL_LICENCE_ACCEPTED=1 first
```

Files land under `Settings.raw_root` (default `data/raw/<source>/...`). Parsing happens in [`clg.parse`](../parse/README.md); loading into Neo4j happens in [`clg.cli` → `clg load ...`](../cli/README.md).

## Licences (read before bulk-fetching)

- **CourtListener** — permissive; be polite on rate limits.
- **legislation.gov.uk** — Open Government Licence v3.0.
- **EUR-Lex** — Commission re-use notice.
- **retsinformation.dk** — Civilstyrelsen open licence.
- **domstol.dk** — open access, attribution required. Free corpus only.
- **Karnov / Ufr** — commercial DK reporters. Ingester is a skeleton until the firm confirms a subscription.
- **Find Case Law** — programmatic bulk requires the free TNA computational-analysis licence. Apply before flipping the env flag.
