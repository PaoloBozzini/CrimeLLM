"""``clg link ...`` — citation + treatment extraction (Phases 1 + 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Citation + treatment extraction (Phase 1/5).", no_args_is_help=True)


@app.command("citations")
def citations(
    text: Annotated[
        str | None,
        typer.Option("--text", help="Raw text to scan for citations."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Read text from this file instead of --text."),
    ] = None,
    jurisdiction: Annotated[
        str | None,
        typer.Option(
            "--jurisdiction",
            "-j",
            help="Restrict to one jurisdiction (US|EW|UK|EU|DK). "
            "Default: all parsers whose code is in enabled_jurisdictions.",
        ),
    ] = None,
) -> None:
    """Run the per-jurisdiction citation parsers over a text blob.

    Dispatches through ``link.cite_registry``. Removing a jurisdiction from
    ``ENABLED_JURISDICTIONS`` (or via ``--jurisdiction``) skips that parser.
    """
    from ..config import get_settings
    from ..link import extract_all, for_jurisdiction, parsers_for_enabled

    if text is None and file is None:
        raise typer.BadParameter("pass --text or --file")
    if text is not None and file is not None:
        raise typer.BadParameter("--text and --file are mutually exclusive")
    body = file.read_text(encoding="utf-8") if file else (text or "")

    settings = get_settings()
    if jurisdiction:
        if not settings.is_enabled(jurisdiction):
            raise typer.BadParameter(
                f"{jurisdiction!r} is not in enabled_jurisdictions={settings.enabled_jurisdictions}"
            )
        parser = for_jurisdiction(jurisdiction)
        if parser is None:
            raise typer.BadParameter(f"no parser registered for {jurisdiction!r}")
        hits = extract_all(body, parsers=[parser])
    else:
        hits = extract_all(body, parsers=parsers_for_enabled(settings))

    typer.echo(
        json.dumps(
            {
                "jurisdictions": sorted({h.jurisdiction for h in hits}),
                "count": len(hits),
                "hits": [
                    {
                        "raw": h.raw,
                        "normalised_id": h.normalised_id,
                        "kind": h.kind,
                        "span": list(h.span),
                        "jurisdiction": h.jurisdiction,
                    }
                    for h in hits
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("distill")
def distill(
    sample: Annotated[
        int,
        typer.Option(
            "--sample",
            "-n",
            help="Number of edges to sample + label with the teacher.",
        ),
    ] = 1000,
    out: Annotated[
        str,
        typer.Option("--out", "-o", help="CSV destination."),
    ] = "data/training/treatment.csv",
    teacher: Annotated[
        str,
        typer.Option(
            "--teacher",
            help="anthropic | ollama. Anthropic = best quality, ~$30 for 50k labels.",
        ),
    ] = "anthropic",
    teacher_model: Annotated[
        str | None,
        typer.Option(
            "--teacher-model",
            help="Override the teacher's model.",
        ),
    ] = None,
    only_with_sentence: Annotated[
        bool,
        typer.Option(
            "--only-with-sentence/--allow-empty-sentence",
            help="Skip edges with no citing_sentence (they teach the student nothing).",
        ),
    ] = True,
    jurisdiction: Annotated[
        str | None,
        typer.Option("--jurisdiction", "-j", help="US|EW|UK|EU|DK filter."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
) -> None:
    """Sample edges from Neo4j, label them with a teacher, write a training CSV.

    The CSV's schema (``text,label,label_str,confidence,teacher,citing_case_id,
    cited_case_id``) drops straight into ``crimellm.classifier.load_dataset_from_csv``.
    """
    from ..link import (
        ClaudeTreatmentClassifier,
        OllamaTreatmentClassifier,
        label_distribution,
        label_with_teacher,
        sample_edges,
        write_training_csv,
    )

    name = teacher.lower()
    if name == "anthropic":
        teacher_clf = ClaudeTreatmentClassifier(
            model=teacher_model or "claude-haiku-4-5-20251001",
        )
    elif name == "ollama":
        teacher_clf = OllamaTreatmentClassifier(
            model=teacher_model or "qwen2.5:14b-instruct",
        )
    else:
        raise typer.BadParameter(f"unknown --teacher {teacher!r}; pick anthropic / ollama")

    edges = list(
        sample_edges(
            n=sample,
            only_with_sentence=only_with_sentence,
            jurisdiction=jurisdiction,
            seed=seed,
        )
    )
    typer.echo(f"sampled {len(edges)} edges")
    if not edges:
        typer.echo("no edges to distill; nothing written")
        raise typer.Exit(code=1)

    samples = label_with_teacher(edges, teacher=teacher_clf, batch_size=batch_size)
    n_written = write_training_csv(samples, out)
    typer.echo(
        json.dumps(
            {
                "sampled": len(edges),
                "labelled": len(samples),
                "written": n_written,
                "out": out,
                "teacher": teacher_clf.name,
                "distribution": label_distribution(samples),
            },
            indent=2,
        )
    )


@app.command("train-distilled")
def train_distilled_cmd(
    csv_in: Annotated[
        str,
        typer.Option(
            "--in",
            "-i",
            help="Path to the training CSV produced by `clg link distill`.",
        ),
    ],
    base_model: Annotated[
        str,
        typer.Option(
            "--base-model",
            help="HF encoder. law-ai/InLegalBERT is the default; "
            "microsoft/deberta-v3-base is a strong general option.",
        ),
    ] = "law-ai/InLegalBERT",
    out: Annotated[
        str,
        typer.Option("--out", "-o", help="Where to save the fine-tuned model."),
    ] = "artifacts/treatment_head",
    epochs: Annotated[int, typer.Option("--epochs")] = 4,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 2e-5,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
    max_len: Annotated[int, typer.Option("--max-len")] = 256,
    test_size: Annotated[float, typer.Option("--test-size")] = 0.15,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    freeze_encoder: Annotated[
        bool,
        typer.Option(
            "--freeze-encoder/--full-finetune",
            help="Head-only trains faster but caps quality. Switch to full "
            "fine-tune when you have 5k+ teacher labels.",
        ),
    ] = False,
) -> None:
    """Fine-tune a HuggingFace encoder on the distilled labels.

    Wraps the existing ``crimellm.classifier.train`` pipeline at
    ``num_labels=10`` with the treatment vocabulary. Reports macro-F1 + per-label
    F1 on the held-out split, plus the path you can hand to the cascade as
    ``--distilled-dir``.
    """
    from ..link import classification_report_text, train_distilled_head

    result = train_distilled_head(
        csv_in,
        base_model=base_model,
        output_dir=out,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_len=max_len,
        test_size=test_size,
        seed=seed,
        freeze_encoder=freeze_encoder,
    )
    typer.echo(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "n_train": result.n_train,
                "n_test": result.n_test,
                "base_model": result.base_model,
                "eval_metrics": result.eval_metrics,
                "per_label_f1": result.per_label_f1,
            },
            indent=2,
            default=str,
        )
    )
    typer.echo("\n" + classification_report_text(result.per_label_f1))
    typer.echo(
        f"\nNext step: `clg link treatment --backend rules+distilled+ollama --distilled-dir {out}`"
    )


def _build_cascade(
    backends: list[str],
    *,
    confidence_threshold: float,
    ollama_model: str | None,
    anthropic_model: str | None,
    distilled_dir: str | None,
    max_claude_edges: int | None,
):
    """Translate the ``--backend rules+distilled+ollama+anthropic`` flag.

    Missing optional deps (the [classifier] / anthropic / ollama bits) cause
    the offending tier to be skipped with a stderr note rather than a hard
    error — Phase 5.3 is meant to run partially when only some tiers are
    available.

    Phase 6: when ``rules`` is requested, one ``RuleTreatmentClassifier``
    is built **per enabled jurisdiction** so DK + EU rule sets coexist
    without one stomping on the other. Each classifier abstains on edges
    from other jurisdictions; the cascade chains them so any matching
    jurisdiction's rules get a shot before escalation.
    """
    from ..config import get_settings
    from ..link import (
        CascadeClassifier,
        ClaudeTreatmentClassifier,
        DistilledTreatmentClassifier,
        OllamaTreatmentClassifier,
        RuleTreatmentClassifier,
    )

    tiers = []
    for name in backends:
        name = name.strip().lower()
        if name == "rules":
            for j in get_settings().enabled_jurisdictions:
                tiers.append(
                    (RuleTreatmentClassifier(jurisdiction=j), confidence_threshold)
                )
        elif name == "distilled":
            if not distilled_dir:
                typer.secho(
                    "[skip] distilled tier: pass --distilled-dir to use it.",
                    err=True,
                    fg="yellow",
                )
                continue
            try:
                tiers.append(
                    (
                        DistilledTreatmentClassifier(model_dir=distilled_dir),
                        confidence_threshold,
                    )
                )
            except (ImportError, FileNotFoundError) as e:
                typer.secho(f"[skip] distilled tier: {e}", err=True, fg="yellow")
        elif name == "ollama":
            tiers.append(
                (
                    OllamaTreatmentClassifier(
                        model=ollama_model or "qwen2.5:7b-instruct",
                    ),
                    confidence_threshold,
                )
            )
        elif name == "anthropic":
            try:
                claude = ClaudeTreatmentClassifier(
                    model=anthropic_model or "claude-haiku-4-5-20251001",
                )
            except (ImportError, RuntimeError) as e:
                typer.secho(f"[skip] anthropic tier: {e}", err=True, fg="yellow")
                continue
            tiers.append((claude, confidence_threshold))
        else:
            raise typer.BadParameter(
                f"unknown backend {name!r}; pick rules/distilled/ollama/anthropic"
            )

    if not tiers:
        raise typer.BadParameter(
            "no tiers active; check --backend and that the relevant extras are installed"
        )

    budget = {"anthropic": max_claude_edges} if max_claude_edges else None
    return CascadeClassifier(tiers, budget_per_tier=budget)


@app.command("treatment")
def treatment(
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help=(
                "Comma-or-plus list, ordered cheapest first. "
                "Pick from rules / distilled / ollama / anthropic."
            ),
        ),
    ] = "rules+ollama",
    confidence_threshold: Annotated[
        float,
        typer.Option(
            "--confidence-threshold",
            "-t",
            help="Minimum confidence for a tier's result to be accepted (else escalate).",
        ),
    ] = 0.85,
    only_with_sentence: Annotated[
        bool,
        typer.Option(
            "--only-with-sentence",
            help="Skip edges that have no citing_sentence (rules + distilled abstain anyway).",
        ),
    ] = False,
    jurisdiction: Annotated[
        str | None,
        typer.Option(
            "--jurisdiction",
            "-j",
            help="Restrict to one jurisdiction (US|EW|UK|EU|DK).",
        ),
    ] = None,
    max_edges: Annotated[
        int | None,
        typer.Option("--max-edges", help="Cap total edges processed this run (resumable)."),
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    ollama_model: Annotated[str | None, typer.Option("--ollama-model")] = None,
    anthropic_model: Annotated[str | None, typer.Option("--anthropic-model")] = None,
    distilled_dir: Annotated[
        str | None,
        typer.Option(
            "--distilled-dir",
            help="Path to the trained distilled head (output of `clg link train-distilled`).",
        ),
    ] = None,
    max_claude_edges: Annotated[
        int | None,
        typer.Option(
            "--max-claude-edges",
            help="Budget cap on Claude escalation tier across this run.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Run the cascade but do not write treatments back to Neo4j.",
        ),
    ] = False,
) -> None:
    """Stream un-classified ``CITES`` edges through the cascade and write back.

    Resumable. Re-running picks up only edges still at ``treatment=neutral``
    (or NULL). Per-tier counts + total written are emitted at the end.
    """
    from ..graph import get_store, iter_neutral_cites, write_treatments
    from ..link import EdgeContext

    backends = [b for b in backend.replace("+", ",").split(",") if b.strip()]
    cascade = _build_cascade(
        backends,
        confidence_threshold=confidence_threshold,
        ollama_model=ollama_model,
        anthropic_model=anthropic_model,
        distilled_dir=distilled_dir,
        max_claude_edges=max_claude_edges,
    )

    store = get_store()
    store.verify()

    by_tier: dict[str, int] = {}
    total = 0
    written = 0

    def _flush(batch_rows: list[dict]) -> None:
        nonlocal written
        nonlocal by_tier
        edges = [
            EdgeContext(
                citing_case_id=r["citing_case_id"],
                cited_case_id=r["cited_case_id"],
                citing_sentence=r.get("citing_sentence", "") or "",
                citing_case_name=r.get("citing_case_name", "") or "",
                cited_case_name=r.get("cited_case_name", "") or "",
                citing_decision_date=str(r.get("citing_decision_date"))
                if r.get("citing_decision_date")
                else None,
                cited_decision_date=str(r.get("cited_decision_date"))
                if r.get("cited_decision_date")
                else None,
                citing_case_jurisdiction=r.get("citing_case_jurisdiction"),
                depth=float(r.get("depth") or 1.0),
            )
            for r in batch_rows
        ]
        report = cascade.classify(edges)
        for tier, count in report.by_tier().items():
            by_tier[tier] = by_tier.get(tier, 0) + count

        if dry_run:
            return

        write_rows = [
            {
                "edge_id": batch_rows[i]["edge_id"],
                "treatment": report.results[i].label,
                "treatment_source": report.results[i].source,
                "treatment_confidence": float(report.results[i].confidence),
            }
            for i in range(len(batch_rows))
        ]
        write_treatments(write_rows, store=store)
        written += len(write_rows)

    pending: list[dict] = []
    for row in iter_neutral_cites(
        only_with_sentence=only_with_sentence,
        jurisdiction=jurisdiction,
        limit=max_edges,
        store=store,
    ):
        pending.append(row)
        total += 1
        if len(pending) >= batch_size:
            _flush(pending)
            pending = []
    if pending:
        _flush(pending)

    typer.echo(
        json.dumps(
            {
                "total_edges": total,
                "written": written,
                "dry_run": dry_run,
                "tier_counts": by_tier,
                "backends": backends,
                "confidence_threshold": confidence_threshold,
            },
            indent=2,
        )
    )
