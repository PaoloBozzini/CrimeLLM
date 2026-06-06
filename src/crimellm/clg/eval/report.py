"""Serialise an ``EvalReport`` into Markdown or JSON.

JSON is the canonical machine-readable form (round-trips through
``json.loads`` cleanly). Markdown is the human view — one section per
metric family + a per-question table.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .runner import EvalReport


def to_dict(report: EvalReport, *, include_answers: bool = False) -> dict:
    payload: dict = {
        "gold_set": {
            "name": report.gold_set_name,
            "version": report.gold_set_version,
            "n_questions": report.n_questions(),
        },
        "embedder": report.embedder_name,
        "synthesizer": report.synthesizer_name,
        "aggregate": asdict(report.aggregate),
        "scores": [asdict(s) for s in report.scores],
    }
    if include_answers:
        payload["answers"] = [a.to_dict() for a in report.answers]
    return payload


def to_json(report: EvalReport, *, indent: int = 2, include_answers: bool = False) -> str:
    return json.dumps(to_dict(report, include_answers=include_answers), indent=indent, default=str)


def _fmt(v: float | None, *, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _bool_mark(v: bool | None) -> str:
    if v is None:
        return "—"
    return "✓" if v else "✗"


def to_markdown(report: EvalReport) -> str:
    agg = report.aggregate
    lines: list[str] = []
    lines.append(f"# {report.gold_set_name} — eval report (v{report.gold_set_version})")
    lines.append("")
    lines.append(f"- **Questions:** {report.n_questions()}")
    lines.append(f"- **Embedder:** {report.embedder_name}")
    lines.append(f"- **Synthesizer:** {report.synthesizer_name}")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| recall@k (mean) | {_fmt(agg.recall_at_k_mean)} |")
    lines.append(
        f"| citation accuracy (mean) | {_fmt(agg.citation_accuracy_mean)} "
        f"({agg.n_with_citation_accuracy} questions) |"
    )
    lines.append(f"| citation accuracy = 1.0 rate | {_fmt_pct(agg.citation_accuracy_perfect)} |")
    lines.append(
        f"| fabrication rate (questions with ≥1 bad cite) | "
        f"{agg.questions_with_fabrication} / {agg.n} "
        f"({agg.fabrication_rate * 100:.1f}%) |"
    )
    lines.append(
        f"| good-law precision (mean) | {_fmt(agg.good_law_precision_mean)} "
        f"({agg.n_with_good_law} questions) |"
    )
    lines.append(f"| good-law recall (mean) | {_fmt(agg.good_law_recall_mean)} |")
    lines.append(f"| good-law treater-named rate | {_fmt_pct(agg.good_law_named_treater_rate)} |")
    lines.append(
        f"| as-of correctness rate | {_fmt_pct(agg.as_of_correct_rate)} ({agg.n_as_of} questions) |"
    )
    lines.append("")
    lines.append("## Per-question")
    lines.append("")
    lines.append(
        "| id | task | recall@k | cite_acc | gl_prec | gl_rec | as_of | named_treater | fabricated |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in report.scores:
        fabricated = ", ".join(s.fabricated_citations) if s.fabricated_citations else ""
        lines.append(
            f"| `{s.question_id}` | {s.task_type} | "
            f"{_fmt(s.recall_at_k)} | {_fmt(s.citation_accuracy)} | "
            f"{_fmt(s.good_law_precision)} | {_fmt(s.good_law_recall)} | "
            f"{_bool_mark(s.as_of_correct)} | "
            f"{_bool_mark(s.good_law_named_treater)} | "
            f"{fabricated} |"
        )
    lines.append("")

    bad = [s for s in report.scores if s.missing_authorities]
    if bad:
        lines.append("## Missing authorities")
        lines.append("")
        for s in bad:
            lines.append(
                f"- **{s.question_id}** ({s.task_type}): missing "
                + ", ".join(f"`{m}`" for m in s.missing_authorities)
            )
        lines.append("")

    return "\n".join(lines)
