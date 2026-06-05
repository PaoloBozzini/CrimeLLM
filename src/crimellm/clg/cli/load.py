"""``clg load ...`` — stream parsed sources into Neo4j."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ..graph import get_store, load_cases, load_citations, load_courts
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
