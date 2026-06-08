"""``clg graph ...`` — schema + admin + gate queries."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from ..graph import (
    apply_schema,
    citation_counts,
    cited_cases,
    citing_cases,
    drop_schema,
    get_store,
    provision_as_of,
    rebuild_vector_index,
    schema_status,
    search_chunks,
)

app = typer.Typer(help="Neo4j schema + admin.", no_args_is_help=True)


@app.command("init")
def init() -> None:
    """Apply constraints, indexes, vector index, jurisdiction seeds."""
    store = get_store()
    store.verify()
    counts = apply_schema(store)
    typer.echo(json.dumps({"applied": counts, "uri": store.settings.neo4j_uri}, indent=2))


@app.command("status")
def status() -> None:
    """Show schema objects + node counts."""
    store = get_store()
    store.verify()
    schema = schema_status(store)
    counts = store.run(
        "MATCH (n) WITH labels(n) AS ls UNWIND ls AS l RETURN l AS label, count(*) AS n "
        "ORDER BY label"
    )
    typer.echo(json.dumps({"schema": schema, "node_counts": counts}, indent=2))


@app.command("wipe")
def wipe(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm DETACH DELETE.")] = False,
) -> None:
    """Delete all nodes + relationships (keeps schema). Requires --yes."""
    if not yes:
        typer.echo("Refusing to wipe without --yes.")
        raise typer.Exit(code=2)
    store = get_store()
    store.run("MATCH (n) DETACH DELETE n")
    typer.echo("wiped")


@app.command("rebuild-vector-index")
def rebuild_vector_index_cmd(
    dim: Annotated[
        int,
        typer.Option(
            "--dim",
            help="New vector dimension (e.g. 4096 for Qwen/Qwen3-Embedding-8B, "
            "1024 for BAAI/bge-m3 or Qwen/Qwen3-Embedding-0.6B, "
            "384 for sentence-transformers/all-MiniLM-L6-v2).",
        ),
    ],
    drop_chunks: Annotated[
        bool,
        typer.Option(
            "--drop-chunks",
            help="Also DETACH DELETE existing Chunk nodes. Required when changing dim "
            "for real, since old embeddings are the wrong size.",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm destructive flags.")] = False,
) -> None:
    """Drop + recreate ``chunk_embedding`` at a new dimension.

    Use after switching embedder backends with a different vector size.
    Pass ``--drop-chunks --yes`` to also remove stale Chunk nodes; otherwise
    the existing chunks stay on disk but become un-queryable.
    """
    if drop_chunks and not yes:
        typer.echo("Refusing to delete Chunk nodes without --yes.")
        raise typer.Exit(code=2)
    out = rebuild_vector_index(dim, drop_chunks=drop_chunks)
    typer.echo(json.dumps(out, indent=2))


@app.command("drop-schema")
def drop_schema_cmd(
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Drop constraints + indexes (data untouched)."""
    if not yes:
        typer.echo("Refusing without --yes.")
        raise typer.Exit(code=2)
    drop_schema(get_store())
    typer.echo("schema dropped")


@app.command("cites")
def cites(
    case_id: Annotated[str, typer.Argument(help="Seed Case node id, e.g. cl-cluster-12345.")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 25,
) -> None:
    """Inbound CITES: cases that cite the seed."""
    typer.echo(json.dumps(citing_cases(case_id, limit=limit), default=str, indent=2))


@app.command("cited-by")
def cited_by(
    case_id: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 25,
) -> None:
    """Outbound CITES: cases the seed cites."""
    typer.echo(json.dumps(cited_cases(case_id, limit=limit), default=str, indent=2))


@app.command("counts")
def counts(
    case_id: Annotated[str, typer.Argument()],
) -> None:
    """Inbound + outbound CITES counts for the seed."""
    typer.echo(json.dumps(citation_counts(case_id), indent=2))


@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Free-text query.")],
    k: Annotated[int, typer.Option("--k", "-k")] = 5,
    jurisdiction: Annotated[
        str | None,
        typer.Option(
            "--jurisdiction",
            "-j",
            help="US|EW|UK|EU|DK filter (must be in ENABLED_JURISDICTIONS).",
        ),
    ] = None,
    parent_type: Annotated[
        str | None,
        typer.Option("--parent-type", help="Case|Provision filter."),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="voyage|openai|sentence-transformers|fake. Default = auto.",
        ),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Override the embedder's model name.")
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="Local backends only: cpu|cuda|mps.")
    ] = None,
) -> None:
    """Vector search over Chunk embeddings, resolved up to parent entity."""
    from ..embed.embedder import get_embedder

    embedder = get_embedder(backend, model=model, device=device)
    qvec = embedder.embed(query)
    rows = search_chunks(
        qvec,
        k=k,
        jurisdiction=jurisdiction,
        parent_type=parent_type,
    )
    typer.echo(json.dumps(rows, default=str, indent=2))


@app.command("provision-as-of")
def provision_as_of_cmd(
    instrument_id: Annotated[
        str,
        typer.Option("--instrument", "-i", help="Instrument id, e.g. uk/ukpga/2006/35."),
    ],
    section: Annotated[str, typer.Option("--section", "-s", help="Section path, e.g. s.1.")],
    as_of: Annotated[str, typer.Option("--as-of", help="ISO date (YYYY-MM-DD).")],
) -> None:
    """Return the Provision text valid on the given date."""
    row = provision_as_of(instrument_id, section, as_of)
    typer.echo(json.dumps(row, default=str, indent=2))
