"""``clg autofetch ...`` — manage the reconciliation queue.

Operator-facing subset of the autofetch subsystem. ``drain`` is the only
verb that needs a configured source registry (which lands in B.6 / Phase C);
``enqueue`` / ``status`` / ``list-pending`` / ``promote`` are queue-only and
shippable now.

Queue path defaults to ``Settings.autofetch_queue_path``; ``--queue-path``
overrides for tests and ops one-offs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from ..autofetch.circuit_breaker import CircuitBreaker
from ..autofetch.quarantine import promote as quarantine_promote
from ..autofetch.queue import SqliteQueue
from ..autofetch.resolver import resolve
from ..config import get_settings
from ..graph.driver import get_store

app = typer.Typer(help="Autofetch queue admin.", no_args_is_help=True)


def _queue(path: Path | None) -> SqliteQueue:
    target = path or get_settings().autofetch_queue_path
    return SqliteQueue(target)


# --- enqueue ---------------------------------------------------------------


@app.command("enqueue")
def enqueue(
    cite_id: Annotated[str, typer.Argument(help="Canonical citation id (ECLI / ELI / CELEX / …)")],
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            help="Override the resolver (e.g. for Karnov-gated ids the resolver skips).",
        ),
    ] = None,
    depth: Annotated[
        int, typer.Option("--depth", help="Cascade depth (0 = operator-triggered)")
    ] = 0,
    queue_path: Annotated[Path | None, typer.Option("--queue-path")] = None,
) -> None:
    """Add a cite id to the autofetch queue."""
    resolved = source or resolve(cite_id)
    if resolved is None:
        typer.echo(
            f"no resolver match for {cite_id!r}; pass --source to force a backend"
        )
        raise typer.Exit(code=2)
    q = _queue(queue_path)
    try:
        created = q.enqueue(cite_id, resolved, depth=depth)
    finally:
        q.close()
    payload = {"cite_id": cite_id, "source": resolved, "created": created, "depth": depth}
    typer.echo(json.dumps(payload, indent=2))


# --- status ----------------------------------------------------------------


@app.command("status")
def status(
    queue_path: Annotated[Path | None, typer.Option("--queue-path")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="text|json")] = "text",
) -> None:
    """Report queue depth, per-source breakdown, recent errors."""
    q = _queue(queue_path)
    try:
        pending = q.list_pending(limit=10_000)
    finally:
        q.close()
    by_source: Counter[str] = Counter(p.source for p in pending)
    by_attempts: Counter[int] = Counter(p.attempts for p in pending)

    # Pull breaker state for every source the queue has rows for, plus the
    # configured default qps map so an operator sees breakers that haven't
    # had any traffic yet too. Best-effort: missing autofetch_circuit table
    # returns "closed" everywhere.
    settings = get_settings()
    circuit_path = queue_path or settings.autofetch_queue_path
    breaker = CircuitBreaker(circuit_path)
    try:
        watched = set(by_source) | set(settings.autofetch_source_qps)
        circuits: dict[str, dict[str, object]] = {}
        for src in sorted(watched):
            row = breaker._read(src)
            circuits[src] = (
                {"state": "closed", "failures": 0}
                if row is None
                else {
                    "state": row["state"],
                    "failures": row["failures"],
                    "next_attempt_at": row["next_attempt_at"],
                }
            )
    finally:
        breaker.close()

    payload = {
        "pending": len(pending),
        "by_source": dict(by_source),
        "by_attempts": dict(by_attempts),
        "recent_errors": [
            {"cite_id": p.cite_id, "error": p.error}
            for p in pending
            if p.error
        ][:10],
        "circuits": circuits,
    }
    if fmt.lower() == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(f"pending: {payload['pending']}")
    if by_source:
        typer.echo("by source:")
        for src, n in by_source.most_common():
            typer.echo(f"  {src}: {n}")
    if payload["recent_errors"]:
        typer.echo("recent errors:")
        for row in payload["recent_errors"]:
            typer.echo(f"  {row['cite_id']}: {row['error']}")


# --- list-pending ----------------------------------------------------------


@app.command("list-pending")
def list_pending(
    queue_path: Annotated[Path | None, typer.Option("--queue-path")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 50,
) -> None:
    """One cite id per line, oldest first."""
    q = _queue(queue_path)
    try:
        rows = q.list_pending(limit=limit)
    finally:
        q.close()
    for r in rows:
        typer.echo(f"{r.cite_id}\t{r.source}\tattempts={r.attempts}\tdepth={r.depth}")


# --- promote ---------------------------------------------------------------


@app.command("promote")
def promote(
    cite_id: Annotated[str, typer.Argument(help="Cite id to flip validated=true.")],
) -> None:
    """Mark an auto-ingested node as human-validated.

    Flips ``validated=true`` on any Case / Provision / Instrument whose ``id``
    matches the given cite. Leaves ``auto_ingested`` alone so the historical
    fact survives — only the eval filter cares about ``validated``.
    """
    store = get_store()
    n = quarantine_promote(cite_id, store=store)
    typer.echo(f"promoted {n} node(s) for {cite_id}")
    raise typer.Exit(code=0)


# --- drain -----------------------------------------------------------------


@app.command("drain")
def drain(
    max_jobs: Annotated[int, typer.Option("--max", help="Cap jobs processed.")] = 20,
    queue_path: Annotated[Path | None, typer.Option("--queue-path")] = None,
) -> None:
    """Run the worker loop up to ``--max`` times (Phase B.6 wires sources)."""
    # Real source-registry wiring lands in B.6 + Phase C. Until then, drain
    # would have nothing to dispatch to — so we exit loudly rather than
    # silently no-op when an operator runs it expecting work to happen.
    typer.echo("drain: source registry not yet wired (Phase B.6 / Phase C)")
    raise typer.Exit(code=2)
