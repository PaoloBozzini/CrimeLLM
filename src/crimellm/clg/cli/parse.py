"""``clg parse ...`` — USLM + Akoma Ntoso parsers (Phase 1+)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

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


@app.command("retsinformation")
def retsinformation(
    file: Annotated[
        Path,
        typer.Option("--file", "-f", help="Cached Retsinformation XML body."),
    ],
    doc_type: Annotated[
        str,
        typer.Option(
            "--doc-type",
            help="DK doc type: lov|lbk|bek|ltc|vejledning.",
        ),
    ],
    year: Annotated[int, typer.Option("--year")],
    num: Annotated[int, typer.Option("--num")],
    explode_subparagraphs: Annotated[
        bool,
        typer.Option(
            "--explode-subparagraphs/--fold-subparagraphs",
            help="True: stk/nr → separate Provisions. False: fold into § text.",
        ),
    ] = True,
) -> None:
    """Parse a single cached Retsinformation XML; print extracted entities as JSON."""
    from ..parse import retsinformation as P

    pr = P.parse_statute_file(
        file,
        doc_type=doc_type,
        year=year,
        num=num,
        explode_subparagraphs=explode_subparagraphs,
    )
    out = {
        "instrument_id": pr.instrument.id,
        "short_title": pr.instrument.short_title,
        "year": pr.instrument.year,
        "provisions": [
            {
                "id": p.id,
                "section_path": p.section_path,
                "valid_from": str(p.valid_from) if p.valid_from else None,
                "text_preview": (p.text or "")[:200],
            }
            for p in pr.provisions
        ],
        "cites_eu_celex": pr.cites_eu_celex,
    }
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False))


@app.command("eurlex")
def eurlex(
    file: Annotated[
        Path,
        typer.Option("--file", "-f", help="Cached EUR-Lex AKN / FORMEX XML body."),
    ],
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help="'regulation' | 'directive' | 'judgment'. Drives which parser runs.",
        ),
    ] = "regulation",
    celex: Annotated[
        str | None,
        typer.Option("--celex", help="Override CELEX (else read from FRBRalias)."),
    ] = None,
    language: Annotated[str, typer.Option("--lang", help="ISO 639-1 language code.")] = "en",
) -> None:
    """Parse a single cached EUR-Lex XML and print the extracted entities as JSON."""
    from ..parse import eurlex as P

    if kind in {"regulation", "directive", "decision", "legislation"}:
        pr = P.parse_regulation_file(file, celex=celex, language=language)
        out = {
            "kind": "legislation",
            "instrument_id": pr.instrument.id,
            "short_title": pr.instrument.short_title,
            "year": pr.instrument.year,
            "provisions": [
                {
                    "id": p.id,
                    "section_path": p.section_path,
                    "valid_from": str(p.valid_from) if p.valid_from else None,
                    "text_preview": (p.text or "")[:200],
                }
                for p in pr.provisions
            ],
            "cites_celex": pr.cites_celex,
        }
    elif kind == "judgment":
        jp = P.parse_judgment_file(file, celex=celex, language=language)
        out = {
            "kind": "judgment",
            "case_id": jp.case.id,
            "name": jp.case.name,
            "decision_date": str(jp.case.decision_date) if jp.case.decision_date else None,
            "court_id": jp.case.court_id,
            "cites_ecli": jp.cites_ecli,
            "cites_celex": jp.cites_celex,
            "body_preview": jp.body_text[:300],
        }
    else:
        raise typer.BadParameter(f"unknown --kind {kind!r}; pick regulation|directive|judgment")
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
