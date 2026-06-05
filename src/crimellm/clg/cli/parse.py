"""``clg parse ...`` — USLM + Akoma Ntoso parsers (Phase 1+)."""

from __future__ import annotations

import typer

from ._common import PENDING

app = typer.Typer(help="USLM + Akoma Ntoso parsers (Phase 1+).", no_args_is_help=True)


@app.command("uslm")
def uslm() -> None:
    typer.echo(PENDING)
    raise typer.Exit(code=1)


@app.command("akoma-ntoso")
def akoma_ntoso() -> None:
    typer.echo(PENDING)
    raise typer.Exit(code=1)
