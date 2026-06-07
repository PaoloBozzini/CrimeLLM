"""``clg ingest ...`` — source downloaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ..config import get_settings
from ..ingest import courtlistener as cl_ingest
from ..parse import courtlistener as cl_parse
from ._common import PENDING, cl_raw_dir

app = typer.Typer(help="Source downloaders (Phase 1+).", no_args_is_help=True)


@app.command("courtlistener")
def courtlistener(
    dump_date: Annotated[str, typer.Option("--date", help="Dump date stamp, e.g. 2024-12-31.")],
    files: Annotated[
        str,
        typer.Option("--files", help="Comma list: courts,dockets,clusters,opinions,citations."),
    ] = "courts,dockets,clusters,opinions,citations",
    dest: Annotated[Path | None, typer.Option("--dest")] = None,
) -> None:
    """Download CourtListener bulk CSV dumps (resumable)."""
    selected = [s.strip() for s in files.split(",") if s.strip()]
    paths = cl_ingest.download_all(dump_date, files=selected, dest_dir=dest)
    typer.echo(json.dumps({k: str(p) for k, p in paths.items()}, indent=2))


@app.command("courtlistener-status")
def courtlistener_status(
    dump_date: Annotated[str, typer.Option("--date")],
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
) -> None:
    """Show which CL bulk files are downloaded + the opinion-cluster sidecar."""
    raw_dir = cl_raw_dir(raw_dir)
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
        "path": str(idx),
        "exists": idx.exists(),
        "size_bytes": idx.stat().st_size if idx.exists() else 0,
    }
    typer.echo(json.dumps(rows, indent=2))


@app.command("courtlistener-index")
def courtlistener_index(
    dump_date: Annotated[str, typer.Option("--date")],
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild even if the sidecar exists.")
    ] = False,
) -> None:
    """One-shot pass over opinions.csv.bz2 -> slim opinion_id,cluster_id sidecar.

    This is the slow step (tens of GB of bz2). Run once per dump; subsequent
    `clg load courtlistener` calls skip the re-read.
    """
    raw_dir = cl_raw_dir(raw_dir)
    op_path = cl_ingest.file_path("opinions", dump_date, raw_dir)
    if not op_path.exists():
        typer.echo(f"missing opinions file: {op_path}")
        raise typer.Exit(code=2)
    idx_path = cl_parse.opinion_cluster_index_path(op_path)
    if idx_path.exists() and not force:
        typer.echo(
            json.dumps(
                {"already_built": str(idx_path), "size_bytes": idx_path.stat().st_size}, indent=2
            )
        )
        return
    out = cl_parse.build_opinion_cluster_index(op_path, dest=idx_path, progress=True)
    typer.echo(json.dumps({"sidecar": str(out), "size_bytes": out.stat().st_size}, indent=2))


@app.command("uscode")
def uscode() -> None:
    """US Code USLM bulk (Phase 1)."""
    typer.echo(PENDING)
    raise typer.Exit(code=1)


@app.command("legislation-uk")
def legislation_uk(
    versions: Annotated[
        str,
        typer.Option(
            "--versions",
            help=(
                "Comma list of version labels. Defaults to 'enacted,current'. "
                "Use 'enacted', 'current', or ISO dates like '2020-01-01'."
            ),
        ),
    ] = "enacted,current",
    statutes: Annotated[
        str | None,
        typer.Option(
            "--statutes",
            help=(
                "Comma list of slash-form Act ids (e.g. 'ukpga/2006/35,ukpga/1968/60'). "
                "Defaults to the UK_CRIMINAL_ACTS bundle."
            ),
        ),
    ] = None,
) -> None:
    """Download UK Acts (whole-act CLML XML) for each requested version."""
    from ..ingest._base import IngestContext
    from ..ingest.legislation_uk import (
        UK_CRIMINAL_ACTS,
        LegislationUKSource,
    )

    vs = tuple(v.strip() for v in versions.split(",") if v.strip())
    if statutes:
        triples = tuple(
            (parts[0], int(parts[1]), int(parts[2]))
            for s in statutes.split(",")
            for parts in [s.strip().split("/")]
        )
    else:
        triples = UK_CRIMINAL_ACTS

    src = LegislationUKSource(statutes=triples, versions=vs)
    paths = src.download(IngestContext())
    typer.echo(json.dumps({k: str(p) for k, p in paths.items()}, indent=2))


@app.command("domstol")
def domstol(
    items: Annotated[
        str,
        typer.Option(
            "--items",
            help="CSV of '<ECLI>|<URL>' pairs (use | as delimiter inside a pair). "
            "Example: 'ECLI:DK:HR:2023:1|https://domstol.dk/.../1.pdf,"
            "ECLI:DK:OLR:2023:42|https://.../42.pdf'.",
        ),
    ],
) -> None:
    """Download DK judgments from domstol.dk by operator-supplied (ECLI, URL) list."""
    s = get_settings()
    if not s.is_enabled("DK"):
        raise typer.BadParameter(
            f"'DK' is not in enabled_jurisdictions={s.enabled_jurisdictions}"
        )
    from ..ingest._base import IngestContext
    from ..ingest.domstol import DomstolSource, JudgmentRef

    refs: list[JudgmentRef] = []
    for entry in items.split(","):
        parts = entry.strip().split("|", 1)
        if len(parts) != 2:
            raise typer.BadParameter(
                f"bad --items entry {entry!r}; want '<ECLI>|<URL>'"
            )
        refs.append(JudgmentRef(ecli=parts[0].strip(), url=parts[1].strip()))

    src = DomstolSource(items=tuple(refs))
    paths = src.download(IngestContext())
    typer.echo(json.dumps({k: str(p) for k, p in paths.items()}, indent=2))


@app.command("karnov")
def karnov() -> None:
    """Commercial DK reporter (Karnov Online) — requires firm subscription."""
    s = get_settings()
    if not s.is_enabled("DK"):
        raise typer.BadParameter(
            f"'DK' is not in enabled_jurisdictions={s.enabled_jurisdictions}"
        )
    if not s.karnov_api_key:
        typer.echo(
            "Refusing: KARNOV_API_KEY not set. The Karnov ingester is a "
            "skeleton — the firm must hold a Karnov Online subscription "
            "before this command can run. See ingest/karnov.py."
        )
        raise typer.Exit(code=2)
    typer.echo(PENDING)
    raise typer.Exit(code=1)


@app.command("retsinformation")
def retsinformation(
    items: Annotated[
        str,
        typer.Option(
            "--items",
            help="CSV of slash-form DK statute ids, e.g. "
            "'lbk/2018/502,lov/2023/1100,bek/2024/42'.",
        ),
    ],
) -> None:
    """Download Danish primary law from retsinformation.dk by ELI."""
    s = get_settings()
    if not s.is_enabled("DK"):
        raise typer.BadParameter(
            f"'DK' is not in enabled_jurisdictions={s.enabled_jurisdictions}"
        )
    from ..ingest._base import IngestContext
    from ..ingest.retsinformation import RetsinformationSource

    triples: list[tuple[str, int, int]] = []
    for s_ in items.split(","):
        parts = s_.strip().split("/")
        if len(parts) != 3:
            raise typer.BadParameter(
                f"bad --items entry {s_!r}; want '<doc_type>/<year>/<num>'"
            )
        triples.append((parts[0], int(parts[1]), int(parts[2])))

    src = RetsinformationSource(items=tuple(triples))
    paths = src.download(IngestContext())
    typer.echo(json.dumps({k: str(p) for k, p in paths.items()}, indent=2))


@app.command("eurlex")
def eurlex(
    celex: Annotated[
        str,
        typer.Option(
            "--celex",
            help="CSV of CELEX ids to fetch, e.g. 32016R0679,32019L0770,62012CJ0131.",
        ),
    ],
    langs: Annotated[
        str,
        typer.Option(
            "--lang",
            help="CSV of ISO 639-1 language codes. Default 'en'. EU bodies "
            "publish in 24 languages; 'da,en' is the firm default.",
        ),
    ] = "en",
    fmt: Annotated[
        str,
        typer.Option(
            "--fmt",
            help="EUR-Lex format param: 'fmx4' (FORMEX) or 'xhtml_akn' "
            "(Akoma Ntoso, where available).",
        ),
    ] = "fmx4",
) -> None:
    """Download EUR-Lex / CELLAR bodies for the given CELEX ids."""
    s = get_settings()
    if not s.is_enabled("EU"):
        raise typer.BadParameter(
            f"'EU' is not in enabled_jurisdictions={s.enabled_jurisdictions}"
        )
    from ..ingest._base import IngestContext
    from ..ingest.eurlex import EurLexSource

    ids = tuple(c.strip() for c in celex.split(",") if c.strip())
    if not ids:
        raise typer.BadParameter("--celex produced no ids")
    languages = tuple(l.strip().lower() for l in langs.split(",") if l.strip())
    src = EurLexSource(celex_ids=ids, languages=languages, fmt=fmt)
    paths = src.download(IngestContext())
    typer.echo(json.dumps({k: str(p) for k, p in paths.items()}, indent=2))


@app.command("find-case-law")
def find_case_law() -> None:
    """TNA Find Case Law judgments — requires computational-analysis licence (Phase 2)."""
    s = get_settings()
    if not s.tna_computational_licence_accepted:
        typer.echo(
            "Refusing: set TNA_COMPUTATIONAL_LICENCE_ACCEPTED=1 after applying for "
            "the (free) Find Case Law computational-analysis licence. "
            "See https://caselaw.nationalarchives.gov.uk/"
        )
        raise typer.Exit(code=2)
    typer.echo(PENDING)
    raise typer.Exit(code=1)
