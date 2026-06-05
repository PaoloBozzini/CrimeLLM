"""``clg link ...`` — citation + treatment extraction (Phases 1 + 5)."""

from __future__ import annotations

import typer

from ._common import PENDING

app = typer.Typer(help="Citation + treatment extraction (Phase 1/5).", no_args_is_help=True)


@app.command("citations")
def citations() -> None:
    """Extract citations via eyecite (Phase 1)."""
    typer.echo(PENDING)
    raise typer.Exit(code=1)


@app.command("treatment")
def treatment() -> None:
    """Classify treatment via the cascade (rules -> distilled -> local LLM -> Claude). Phase 5."""
    typer.echo(PENDING)
    raise typer.Exit(code=1)
