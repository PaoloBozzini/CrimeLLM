"""``clg load ...`` — stream parsed sources into Neo4j."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ..graph import (
    get_store,
    load_cases,
    load_citations,
    load_courts,
)
from ..ingest import courtlistener as cl_ingest
from ..parse import courtlistener as cl_parse
from ._common import cl_raw_dir

app = typer.Typer(help="Stream parsed sources into Neo4j.", no_args_is_help=True)


@app.command("courtlistener")
def courtlistener(
    dump_date: Annotated[
        str, typer.Option("--date", help="Dump date stamp matching the downloaded files.")
    ],
    raw_dir: Annotated[
        Path | None,
        typer.Option(
            "--raw-dir", help="Where the bulk CSVs live. Defaults to data/raw/courtlistener."
        ),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Take only the first N clusters (dev slice).")
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 5000,
    skip_citations: Annotated[
        bool, typer.Option("--skip-citations", help="Load courts+cases only.")
    ] = False,
    auto_index: Annotated[
        bool,
        typer.Option(
            "--auto-index/--no-auto-index",
            help="Build the opinion-cluster sidecar if missing.",
        ),
    ] = True,
) -> None:
    """Parse CL bulk CSVs and MERGE into Neo4j (courts, cases, DECIDED, CITES).

    Uses the opinion-cluster sidecar (built via `clg ingest courtlistener-index`)
    so the slow opinions.csv.bz2 pass happens at most once per dump.
    """
    raw_dir = cl_raw_dir(raw_dir)

    def fp(key: str) -> Path:
        return cl_ingest.file_path(key, dump_date, raw_dir)

    store = get_store()
    store.verify()

    typer.echo(">>> courts")
    n_courts = load_courts(
        cl_parse.iter_courts(fp("courts"), progress=True),
        batch_size=1000,
        store=store,
    )

    clusters_csv = fp("clusters")
    typer.echo(">>> clusters: scoping")
    allowed = cl_parse.cluster_ids(clusters_csv, limit=limit, progress=True)
    if limit:
        typer.echo(f"limit={limit} -> {len(allowed)} clusters in scope")

    docket_scope = (
        cl_parse.docket_ids_for_clusters(clusters_csv, allowed, progress=True) if limit else None
    )
    docket_to_court: dict[str, str] = {}
    if fp("dockets").exists():
        typer.echo(">>> dockets -> court")
        docket_to_court = cl_parse.build_docket_to_court(
            fp("dockets"),
            allowed_docket_ids=docket_scope,
            progress=True,
        )

    typer.echo(">>> cases")
    cases_iter = cl_parse.iter_cases(
        clusters_csv,
        docket_to_court=docket_to_court,
        limit=limit,
        progress=True,
    )
    n_cases = load_cases(cases_iter, batch_size=batch_size, store=store)

    n_cites = 0
    if not skip_citations:
        op_path = fp("opinions")
        idx_path = cl_parse.opinion_cluster_index_path(op_path)
        if not idx_path.exists():
            if not auto_index:
                typer.echo(
                    f"opinion-cluster sidecar missing ({idx_path}). "
                    "Run `clg ingest courtlistener-index --date ...` first, "
                    "or re-run with --auto-index."
                )
                raise typer.Exit(code=2)
            typer.echo(">>> building opinion-cluster sidecar (one-time, slow)")
            cl_parse.build_opinion_cluster_index(op_path, dest=idx_path, progress=True)
        typer.echo(">>> loading opinion-cluster sidecar")
        op_to_cluster = cl_parse.load_opinion_cluster_index(
            idx_path,
            allowed_clusters=allowed if limit else None,
            progress=True,
        )
        typer.echo(f"opinion-cluster map size: {len(op_to_cluster):,}")
        typer.echo(">>> citations")
        cites_iter = cl_parse.iter_citations(fp("citations"), op_to_cluster, progress=True)
        n_cites = load_citations(cites_iter, batch_size=10000, store=store)

    typer.echo(
        json.dumps(
            {
                "courts": n_courts,
                "cases": n_cases,
                "citations": n_cites,
                "skipped_citations": skip_citations,
            },
            indent=2,
        )
    )


@app.command("domstol")
def domstol(
    items: Annotated[
        str,
        typer.Option(
            "--items",
            help="CSV of '<ECLI>|<URL>' pairs matching `clg ingest domstol`.",
        ),
    ],
) -> None:
    """Parse cached DK judgments + MERGE into Neo4j (Courts, Cases, citation hits)."""
    from ..config import get_settings
    from ..ingest._base import IngestContext
    from ..ingest.domstol import DomstolSource, JudgmentRef

    s = get_settings()
    if not s.is_enabled("DK"):
        raise typer.BadParameter(
            f"'DK' is not in enabled_jurisdictions={s.enabled_jurisdictions}"
        )

    refs: list[JudgmentRef] = []
    for entry in items.split(","):
        parts = entry.strip().split("|", 1)
        if len(parts) != 2:
            raise typer.BadParameter(
                f"bad --items entry {entry!r}; want '<ECLI>|<URL>'"
            )
        refs.append(JudgmentRef(ecli=parts[0].strip(), url=parts[1].strip()))

    store = get_store()
    store.verify()
    src = DomstolSource(items=tuple(refs))
    ctx = IngestContext(store=store)
    report = src.load(ctx)
    typer.echo(json.dumps({"source": report.source, **report.counts, **report.extras}, indent=2))


@app.command("retsinformation")
def retsinformation(
    items: Annotated[
        str,
        typer.Option(
            "--items",
            help="CSV of accession numbers (e.g. 'A20180050229,B20260050805'). "
            "Each must be downloaded via `clg ingest retsinformation --items ...` first.",
        ),
    ],
    explode_subparagraphs: Annotated[
        bool,
        typer.Option(
            "--explode-subparagraphs/--fold-subparagraphs",
            help="True (default): stk/nr become separate Provision nodes for "
            "finer retrieval. False: fold into the parent § text.",
        ),
    ] = True,
) -> None:
    """Parse cached Retsinformation XML + MERGE into Neo4j (Instruments, Provisions, IMPLEMENTS)."""
    from ..config import get_settings
    from ..ingest._base import IngestContext
    from ..ingest.retsinformation import RetsinformationSource

    s = get_settings()
    if not s.is_enabled("DK"):
        raise typer.BadParameter(
            f"'DK' is not in enabled_jurisdictions={s.enabled_jurisdictions}"
        )

    accns = tuple(a.strip() for a in items.split(",") if a.strip())
    if not accns:
        raise typer.BadParameter("--items produced no accession numbers")

    store = get_store()
    store.verify()
    src = RetsinformationSource(
        accns=accns,
        explode_subparagraphs=explode_subparagraphs,
    )
    ctx = IngestContext(store=store)
    report = src.load(ctx)
    typer.echo(json.dumps({"source": report.source, **report.counts, **report.extras}, indent=2))


@app.command("eurlex")
def eurlex(
    celex: Annotated[
        str,
        typer.Option(
            "--celex",
            help="CSV of CELEX ids to load. Each must be downloaded already.",
        ),
    ],
    langs: Annotated[
        str,
        typer.Option("--lang", help="CSV of language codes matching --lang at ingest."),
    ] = "en",
    fmt: Annotated[
        str, typer.Option("--fmt", help="EUR-Lex format param.")
    ] = "fmx4",
) -> None:
    """Parse cached EUR-Lex XML + MERGE into Neo4j (Instrument, Provision, Case, IMPLEMENTS)."""
    from ..config import get_settings
    from ..ingest._base import IngestContext
    from ..ingest.eurlex import EurLexSource

    s = get_settings()
    if not s.is_enabled("EU"):
        raise typer.BadParameter(
            f"'EU' is not in enabled_jurisdictions={s.enabled_jurisdictions}"
        )

    ids = tuple(c.strip() for c in celex.split(",") if c.strip())
    if not ids:
        raise typer.BadParameter("--celex produced no ids")
    languages = tuple(l.strip().lower() for l in langs.split(",") if l.strip())

    store = get_store()
    store.verify()
    src = EurLexSource(celex_ids=ids, languages=languages, fmt=fmt)
    ctx = IngestContext(store=store)
    report = src.load(ctx)
    typer.echo(json.dumps({"source": report.source, **report.counts, **report.extras}, indent=2))


@app.command("legislation-uk")
def legislation_uk(
    versions: Annotated[
        str, typer.Option("--versions", help="Comma list, e.g. 'enacted,current,2020-01-01'.")
    ] = "enacted,current",
    statutes: Annotated[
        str | None,
        typer.Option("--statutes", help="Comma list of slash-form Act ids."),
    ] = None,
) -> None:
    """Parse cached legislation.gov.uk XML and MERGE into Neo4j.

    Run ``clg ingest legislation-uk --versions ...`` first to fetch the XML.
    Each Act becomes one Instrument node; each section becomes one Provision
    node per version, with ``valid_from`` set so the temporal as-of query
    (``clg graph provision-as-of``) returns the right text for any date.
    """
    from ..ingest._base import IngestContext
    from ..ingest.legislation_uk import UK_CRIMINAL_ACTS, LegislationUKSource

    vs = tuple(v.strip() for v in versions.split(",") if v.strip())
    if statutes:
        triples = tuple(
            (parts[0], int(parts[1]), int(parts[2]))
            for s in statutes.split(",")
            for parts in [s.strip().split("/")]
        )
    else:
        triples = UK_CRIMINAL_ACTS

    store = get_store()
    store.verify()
    src = LegislationUKSource(statutes=triples, versions=vs)
    ctx = IngestContext(store=store)
    report = src.load(ctx)
    typer.echo(json.dumps({"source": report.source, **report.counts, **report.extras}, indent=2))
