"""clg — Common Legal Graph CLI.

    clg graph init               # apply constraints + vector index
    clg graph status             # show schema + counts
    clg graph wipe               # DETACH DELETE everything (with --yes)
    clg graph cites <case_id>    # cases citing the seed (gate query)
    clg graph cited-by <case_id> # cases the seed cites
    clg graph counts <case_id>   # inbound/outbound CITES counts

    clg ingest courtlistener --date YYYY-MM-DD
    clg parse  courtlistener --date YYYY-MM-DD [--limit N] -> JSONL
    clg load   courtlistener --date YYYY-MM-DD [--limit N] -> Neo4j

    clg embed                    # Phase 3
    clg query  "..."             # Phase 4
    clg eval                     # Phase 6

Run `docker compose up -d neo4j` first.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .config import get_settings
from .graph import (
    apply_schema, citation_counts, cited_cases, citing_cases,
    drop_schema, get_store, load_cases, load_citations, load_courts, schema_status,
)
from .ingest import courtlistener as cl_ingest
from .parse import courtlistener as cl_parse

app = typer.Typer(
    name="clg",
    help="Common Legal Graph — Neo4j RAG over US + UK primary law.",
    no_args_is_help=True,
    add_completion=False,
)

graph_app = typer.Typer(help="Neo4j schema + admin.", no_args_is_help=True)
ingest_app = typer.Typer(help="Source downloaders (Phase 1+).", no_args_is_help=True)
parse_app = typer.Typer(help="USLM + Akoma Ntoso parsers (Phase 1+).", no_args_is_help=True)
link_app = typer.Typer(help="Citation + treatment extraction (Phase 1/5).", no_args_is_help=True)

load_app = typer.Typer(help="Stream parsed sources into Neo4j.", no_args_is_help=True)

app.add_typer(graph_app, name="graph")
app.add_typer(ingest_app, name="ingest")
app.add_typer(parse_app, name="parse")
app.add_typer(link_app, name="link")
app.add_typer(load_app, name="load")


# --- graph ------------------------------------------------------------------

@graph_app.command("init")
def graph_init() -> None:
    """Apply constraints, indexes, vector index, jurisdiction seeds."""
    store = get_store()
    store.verify()
    counts = apply_schema(store)
    typer.echo(json.dumps({"applied": counts, "uri": store.settings.neo4j_uri}, indent=2))


@graph_app.command("status")
def graph_status() -> None:
    """Show schema objects + node counts."""
    store = get_store()
    store.verify()
    schema = schema_status(store)
    counts = store.run(
        "MATCH (n) WITH labels(n) AS ls UNWIND ls AS l RETURN l AS label, count(*) AS n "
        "ORDER BY label"
    )
    typer.echo(json.dumps({"schema": schema, "node_counts": counts}, indent=2))


@graph_app.command("wipe")
def graph_wipe(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm DETACH DELETE.")] = False,
) -> None:
    """Delete all nodes + relationships (keeps schema). Requires --yes."""
    if not yes:
        typer.echo("Refusing to wipe without --yes.")
        raise typer.Exit(code=2)
    store = get_store()
    store.run("MATCH (n) DETACH DELETE n")
    typer.echo("wiped")


@graph_app.command("drop-schema")
def graph_drop_schema(
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Drop constraints + indexes (data untouched)."""
    if not yes:
        typer.echo("Refusing without --yes.")
        raise typer.Exit(code=2)
    drop_schema(get_store())
    typer.echo("schema dropped")


@graph_app.command("cites")
def graph_cites(
    case_id: Annotated[str, typer.Argument(help="Seed Case node id, e.g. cl-cluster-12345.")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 25,
) -> None:
    """Inbound CITES: cases that cite the seed."""
    typer.echo(json.dumps(citing_cases(case_id, limit=limit), default=str, indent=2))


@graph_app.command("cited-by")
def graph_cited_by(
    case_id: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 25,
) -> None:
    """Outbound CITES: cases the seed cites."""
    typer.echo(json.dumps(cited_cases(case_id, limit=limit), default=str, indent=2))


@graph_app.command("counts")
def graph_counts(
    case_id: Annotated[str, typer.Argument()],
) -> None:
    """Inbound + outbound CITES counts for the seed."""
    typer.echo(json.dumps(citation_counts(case_id), indent=2))


# --- ingest / parse / link / embed / query / eval stubs ---------------------

_PENDING = "Phase 0 stub — not implemented yet."


@ingest_app.command("courtlistener")
def ingest_courtlistener(
    dump_date: Annotated[str, typer.Option("--date", help="Dump date stamp, e.g. 2024-12-31.")],
    files: Annotated[str, typer.Option("--files", help="Comma list: courts,dockets,clusters,opinions,citations.")] = "courts,dockets,clusters,opinions,citations",
    dest: Annotated[Path | None, typer.Option("--dest")] = None,
) -> None:
    """Download CourtListener bulk CSV dumps (resumable)."""
    selected = [s.strip() for s in files.split(",") if s.strip()]
    paths = cl_ingest.download_all(dump_date, files=selected, dest_dir=dest)
    typer.echo(json.dumps({k: str(p) for k, p in paths.items()}, indent=2))


@ingest_app.command("courtlistener-status")
def ingest_courtlistener_status(
    dump_date: Annotated[str, typer.Option("--date")],
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
) -> None:
    """Show which CL bulk files are downloaded + the opinion-cluster sidecar."""
    raw_dir = raw_dir or (get_settings().raw_root / "courtlistener")
    rows: dict[str, dict[str, object]] = {}
    for key in cl_ingest.BULK_FILES:
        p = cl_ingest.file_path(key, dump_date, raw_dir)
        rows[key] = {
            "path": str(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
        }
    op_path = cl_ingest.file_path("opinions", dump_date, raw_dir)
    idx = cl_parse.opinion_cluster_index_path(op_path)
    rows["opcluster_index"] = {
        "path": str(idx), "exists": idx.exists(),
        "size_bytes": idx.stat().st_size if idx.exists() else 0,
    }
    typer.echo(json.dumps(rows, indent=2))


@ingest_app.command("courtlistener-index")
def ingest_courtlistener_index(
    dump_date: Annotated[str, typer.Option("--date")],
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if the sidecar exists.")] = False,
) -> None:
    """One-shot pass over opinions.csv.bz2 -> slim opinion_id,cluster_id sidecar.

    This is the slow step (tens of GB of bz2). Run once per dump; subsequent
    `clg load courtlistener` calls skip the re-read.
    """
    raw_dir = raw_dir or (get_settings().raw_root / "courtlistener")
    op_path = cl_ingest.file_path("opinions", dump_date, raw_dir)
    if not op_path.exists():
        typer.echo(f"missing opinions file: {op_path}")
        raise typer.Exit(code=2)
    idx_path = cl_parse.opinion_cluster_index_path(op_path)
    if idx_path.exists() and not force:
        typer.echo(json.dumps({"already_built": str(idx_path),
                               "size_bytes": idx_path.stat().st_size}, indent=2))
        return
    out = cl_parse.build_opinion_cluster_index(op_path, dest=idx_path, progress=True)
    typer.echo(json.dumps({"sidecar": str(out),
                           "size_bytes": out.stat().st_size}, indent=2))


@ingest_app.command("uscode")
def ingest_uscode() -> None:
    """US Code USLM bulk (Phase 1)."""
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


@ingest_app.command("legislation-uk")
def ingest_legislation_uk() -> None:
    """legislation.gov.uk Acts + point-in-time versions (Phase 2)."""
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


@ingest_app.command("find-case-law")
def ingest_find_case_law() -> None:
    """TNA Find Case Law judgments — requires computational-analysis licence (Phase 2)."""
    s = get_settings()
    if not s.tna_computational_licence_accepted:
        typer.echo(
            "Refusing: set TNA_COMPUTATIONAL_LICENCE_ACCEPTED=1 after applying for "
            "the (free) Find Case Law computational-analysis licence. "
            "See https://caselaw.nationalarchives.gov.uk/"
        )
        raise typer.Exit(code=2)
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


@parse_app.command("uslm")
def parse_uslm() -> None:
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


@parse_app.command("akoma-ntoso")
def parse_akn() -> None:
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


@load_app.command("courtlistener")
def load_courtlistener(
    dump_date: Annotated[str, typer.Option("--date", help="Dump date stamp matching the downloaded files.")],
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir", help="Where the bulk CSVs live. Defaults to data/raw/courtlistener.")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Take only the first N clusters (dev slice).")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 5000,
    skip_citations: Annotated[bool, typer.Option("--skip-citations", help="Load courts+cases only.")] = False,
    auto_index: Annotated[bool, typer.Option("--auto-index/--no-auto-index", help="Build the opinion-cluster sidecar if missing.")] = True,
) -> None:
    """Parse CL bulk CSVs and MERGE into Neo4j (courts, cases, DECIDED, CITES).

    Uses the opinion-cluster sidecar (built via `clg ingest courtlistener-index`)
    so the slow opinions.csv.bz2 pass happens at most once per dump.
    """
    raw_dir = raw_dir or (get_settings().raw_root / "courtlistener")
    fp = lambda key: cl_ingest.file_path(key, dump_date, raw_dir)

    store = get_store()
    store.verify()

    typer.echo(">>> courts")
    n_courts = load_courts(
        cl_parse.iter_courts(fp("courts"), progress=True),
        batch_size=1000, store=store,
    )

    clusters_csv = fp("clusters")
    typer.echo(">>> clusters: scoping")
    allowed = cl_parse.cluster_ids(clusters_csv, limit=limit, progress=True)
    if limit:
        typer.echo(f"limit={limit} -> {len(allowed)} clusters in scope")

    docket_scope = (
        cl_parse.docket_ids_for_clusters(clusters_csv, allowed, progress=True)
        if limit else None
    )
    docket_to_court: dict[str, str] = {}
    if fp("dockets").exists():
        typer.echo(">>> dockets -> court")
        docket_to_court = cl_parse.build_docket_to_court(
            fp("dockets"), allowed_docket_ids=docket_scope, progress=True,
        )

    typer.echo(">>> cases")
    cases_iter = cl_parse.iter_cases(
        clusters_csv, docket_to_court=docket_to_court, limit=limit, progress=True,
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
            idx_path, allowed_clusters=allowed if limit else None, progress=True,
        )
        typer.echo(f"opinion-cluster map size: {len(op_to_cluster):,}")
        typer.echo(">>> citations")
        cites_iter = cl_parse.iter_citations(fp("citations"), op_to_cluster, progress=True)
        n_cites = load_citations(cites_iter, batch_size=10000, store=store)

    typer.echo(json.dumps({
        "courts": n_courts, "cases": n_cases, "citations": n_cites,
        "skipped_citations": skip_citations,
    }, indent=2))


@link_app.command("citations")
def link_citations() -> None:
    """Extract citations via eyecite (Phase 1)."""
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


@link_app.command("treatment")
def link_treatment() -> None:
    """Classify treatment via Claude over citing sentences (Phase 5)."""
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


@app.command("embed")
def embed_cmd() -> None:
    """Chunk + embed (Phase 3)."""
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


@app.command("query")
def query_cmd(
    question: Annotated[str, typer.Argument(help="Question to ask.")],
    jurisdiction: Annotated[str | None, typer.Option("--jurisdiction", "-j")] = None,
    as_of: Annotated[str | None, typer.Option("--as-of", help="ISO date, default today.")] = None,
) -> None:
    """Grounded answer via graph traversal (Phase 4)."""
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


@app.command("eval")
def eval_cmd() -> None:
    """Run gold-set evaluation (Phase 6)."""
    typer.echo(_PENDING)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
