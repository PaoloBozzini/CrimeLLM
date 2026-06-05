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
    schema_status,
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
