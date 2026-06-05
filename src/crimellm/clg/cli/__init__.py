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

Run ``docker compose up -d neo4j`` first. Each sub-app lives in its own
module under ``clg/cli/`` to keep file sizes bounded as phases land.
"""

from __future__ import annotations

from typing import Annotated

import typer

from . import graph as graph_cli
from . import ingest as ingest_cli
from . import link as link_cli
from . import load as load_cli
from . import parse as parse_cli
from ._common import PENDING

app = typer.Typer(
    name="clg",
    help="Common Legal Graph — Neo4j RAG over US + UK primary law.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(graph_cli.app, name="graph")
app.add_typer(ingest_cli.app, name="ingest")
app.add_typer(parse_cli.app, name="parse")
app.add_typer(link_cli.app, name="link")
app.add_typer(load_cli.app, name="load")


# --- top-level stubs (Phase 3/4/6) -----------------------------------------


@app.command("embed")
def embed_cmd() -> None:
    """Chunk + embed (Phase 3)."""
    typer.echo(PENDING)
    raise typer.Exit(code=1)


@app.command("query")
def query_cmd(
    question: Annotated[str, typer.Argument(help="Question to ask.")],
    jurisdiction: Annotated[str | None, typer.Option("--jurisdiction", "-j")] = None,
    as_of: Annotated[str | None, typer.Option("--as-of", help="ISO date, default today.")] = None,
) -> None:
    """Grounded answer via graph traversal (Phase 4)."""
    typer.echo(PENDING)
    raise typer.Exit(code=1)


@app.command("eval")
def eval_cmd() -> None:
    """Run gold-set evaluation (Phase 6)."""
    typer.echo(PENDING)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()


__all__ = ["app"]
