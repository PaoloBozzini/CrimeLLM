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


# --- top-level commands ----------------------------------------------------


@app.command("embed")
def embed_cmd(
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="voyage|openai|sentence-transformers|fake. Defaults to voyage if VOYAGE_API_KEY is set, else fake.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Override the embedder's model (e.g. 'sentence-transformers/all-mpnet-base-v2').",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Local backends only: 'cpu' / 'cuda' / 'mps'. Defaults to auto.",
        ),
    ] = None,
    parent_type: Annotated[
        str,
        typer.Option(
            "--parent-type",
            help="Provision|Case|all (default).",
        ),
    ] = "all",
    jurisdiction: Annotated[
        str | None,
        typer.Option("--jurisdiction", "-j", help="Restrict to one jurisdiction code (US/EW/UK)."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Cap on entities processed (dev slice).")
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 64,
) -> None:
    """Chunk un-embedded entities + embed + MERGE Chunk nodes (Phase 3)."""
    import json as _json

    from ..embed.chunker import chunk_provision
    from ..embed.embedder import embed_in_batches, get_embedder
    from ..graph import get_store, load_chunks
    from ..models import Provision

    embedder = get_embedder(backend, model=model, device=device)
    store = get_store()
    store.verify()

    # Provision-only path covers the Phase 3 gate. Case bodies come from the
    # judgment XML and are loaded by Phase 4's retrieval slice.
    if parent_type not in {"Provision", "all"}:
        typer.echo("Only Provision chunks are wired in Phase 3.")
        raise typer.Exit(code=2)

    where = "WHERE NOT EXISTS { (:Chunk)-[:PART_OF]->(p) }"
    if jurisdiction:
        where += f" AND p.jurisdiction = '{jurisdiction}'"
    cypher = (
        f"MATCH (p:Provision) {where} "
        "RETURN p.id AS id, p.instrument_id AS instrument_id, "
        "       p.jurisdiction AS jurisdiction, p.section_path AS section_path, "
        "       p.text AS text, p.version_id AS version_id, "
        "       p.valid_from AS valid_from, p.valid_to AS valid_to"
    )
    if limit:
        cypher += f" LIMIT {limit}"

    rows = store.run(cypher)
    typer.echo(f"un-embedded Provisions: {len(rows)}")

    chunks_total = 0
    texts_buf: list[str] = []
    chunks_buf: list = []  # noqa: ANN401
    for r in rows:
        prov = Provision(
            id=r["id"],
            instrument_id=r["instrument_id"],
            jurisdiction=r["jurisdiction"],
            section_path=r["section_path"],
            text=r["text"] or "",
            valid_from=r["valid_from"],
            valid_to=r["valid_to"],
            version_id=r["version_id"],
        )
        for ch in chunk_provision(prov):
            chunks_buf.append(ch)
            texts_buf.append(ch.text)

    if not chunks_buf:
        typer.echo("nothing to embed")
        return

    typer.echo(f"embedding {len(chunks_buf)} chunks via {embedder.name} (dim={embedder.dim})")
    vectors = embed_in_batches(embedder, texts_buf, batch_size=batch_size)
    for ch, vec in zip(chunks_buf, vectors, strict=True):
        ch.embedding = vec

    chunks_total = load_chunks(
        chunks_buf, embedding_model=embedder.name, batch_size=batch_size, store=store
    )
    typer.echo(_json.dumps({"chunks": chunks_total, "model": embedder.name}, indent=2))


@app.command("query")
def query_cmd(
    question: Annotated[str, typer.Argument(help="Question to ask.")],
    jurisdiction: Annotated[
        str | None,
        typer.Option(
            "--jurisdiction",
            "-j",
            help="US|EW|UK. Default = infer from the question.",
        ),
    ] = None,
    as_of: Annotated[str | None, typer.Option("--as-of", help="ISO date. Default = today.")] = None,
    seed_k: Annotated[int, typer.Option("--seed-k", help="Vector-search seeds.")] = 8,
    top_k: Annotated[int, typer.Option("--top-k", help="Candidates kept after rerank.")] = 6,
    embedder_backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Embedder backend: voyage|openai|sentence-transformers|fake.",
        ),
    ] = None,
    synthesizer: Annotated[
        str | None,
        typer.Option(
            "--synth",
            help=(
                "Synthesizer: anthropic|ollama|airllm|fake. "
                "Default: anthropic if ANTHROPIC_API_KEY set, else ollama "
                "if its server is reachable, else fake."
            ),
        ),
    ] = None,
    synth_model: Annotated[
        str | None,
        typer.Option(
            "--synth-model",
            help=(
                "Synthesizer model override (e.g. 'qwen2.5:14b-instruct' for ollama, "
                "'Qwen/Qwen2.5-7B-Instruct' for airllm, 'claude-opus-4-7' for anthropic)."
            ),
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the Answer as JSON (text + citations + caveats + used).",
        ),
    ] = False,
) -> None:
    """Grounded answer via graph traversal (Phase 4).

    Flow: parse question -> seed by vector search -> traverse the graph
    (cited/citing, INTERPRETS, as-of-date Provision versions) -> good-law
    check on Cases -> rerank -> synthesise with strict citation discipline.

    When ``--json`` is passed you get the full structured Answer; otherwise
    plain text + bulleted caveats + the list of used identifiers.
    """
    import json as _json

    from ..retrieval import run_query
    from ..retrieval.synthesize import get_synthesizer

    # Build the synthesizer up-front so --synth-model is honoured.
    synth = get_synthesizer(synthesizer, model=synth_model)

    answer = run_query(
        question,
        jurisdiction=jurisdiction,  # type: ignore[arg-type]
        as_of=as_of,
        seed_k=seed_k,
        top_k=top_k,
        embedder_backend=embedder_backend,
        synthesizer=synth,
    )

    if json_out:
        typer.echo(_json.dumps(answer.to_dict(), default=str, indent=2))
        return

    typer.echo(answer.text)
    if answer.caveats:
        typer.echo("\nCaveats:")
        for cv in answer.caveats:
            typer.echo(f"  - {cv}")
    if answer.citations:
        typer.echo("\nCited:")
        for c in answer.citations:
            typer.echo(f"  - {c}")


@app.command("eval")
def eval_cmd() -> None:
    """Run gold-set evaluation (Phase 6)."""
    typer.echo(PENDING)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()


__all__ = ["app"]
