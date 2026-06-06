"""Run a gold set through the retrieval pipeline + score it.

Pure orchestration. The hard work lives in:

* ``clg.retrieval.run_query`` — produces an ``Answer``.
* ``clg.eval.metrics.score_question`` — turns ``(GoldQuestion, Answer)``
  into a ``QuestionScore``.
* ``clg.eval.metrics.aggregate`` — collapses scores to one summary row.

The runner returns an ``EvalReport`` you can hand to ``report.to_markdown``
or ``report.to_json``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..embed.embedder import Embedder
from ..graph.driver import Neo4jStore
from ..retrieval.query import run_query
from ..retrieval.synthesize import Answer, Synthesizer
from .metrics import AggregateScores, QuestionScore, aggregate, score_question
from .schema import GoldQuestion, GoldSet


@dataclass(slots=True)
class EvalReport:
    """Outcome of an eval run."""

    gold_set_name: str
    gold_set_version: str
    embedder_name: str
    synthesizer_name: str
    scores: list[QuestionScore]
    aggregate: AggregateScores
    answers: list[Answer] = field(default_factory=list)

    def n_questions(self) -> int:
        return len(self.scores)


def _resolve_as_of(gold_q: GoldQuestion) -> str | None:
    if gold_q.as_of is None:
        return None
    return gold_q.as_of.isoformat()


def run_eval(
    gold: GoldSet | Iterable[GoldQuestion],
    *,
    embedder: Embedder | None = None,
    synthesizer: Synthesizer | None = None,
    store: Neo4jStore | None = None,
    seed_k: int = 8,
    top_k: int = 6,
    keep_answers: bool = True,
) -> EvalReport:
    """Score every question in ``gold`` end-to-end."""
    if isinstance(gold, GoldSet):
        questions = list(gold.questions)
        gold_name = gold.name
        gold_version = gold.version
    else:
        questions = list(gold)
        gold_name = "ad-hoc"
        gold_version = "1"

    scores: list[QuestionScore] = []
    answers: list[Answer] = []
    embedder_name = embedder.name if embedder is not None else "<auto>"
    synthesizer_name = synthesizer.name if synthesizer is not None else "<auto>"

    for q in questions:
        ans = run_query(
            q.question,
            jurisdiction=q.jurisdiction,  # type: ignore[arg-type]
            as_of=_resolve_as_of(q),
            seed_k=seed_k,
            top_k=top_k,
            embedder=embedder,
            synthesizer=synthesizer,
            store=store,
        )
        scores.append(score_question(q, ans))
        if keep_answers:
            answers.append(ans)

    return EvalReport(
        gold_set_name=gold_name,
        gold_set_version=gold_version,
        embedder_name=embedder_name,
        synthesizer_name=synthesizer_name,
        scores=scores,
        aggregate=aggregate(scores),
        answers=answers,
    )
