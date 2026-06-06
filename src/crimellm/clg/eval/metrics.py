"""Deterministic, identifier-level metrics over a single answer.

Brief Phase 6 — four families:

1. **Retrieval recall@k** — fraction of expected authorities that appear in
   the top-k candidates the retriever surfaced.
2. **Citation accuracy** — fraction of cited identifiers in the answer text
   that are valid (i.e. exist in the retrieved context). Zero tolerance for
   fabrication: even one bad cite drops this to ``< 1.0``.
3. **Good-law precision/recall** — for cases with annotated
   ``expected_good_law``, did we flag them correctly and name the treating
   case?
4. **As-of correctness** — for ``as_of_date`` questions, did the system
   surface the expected Provision *version* (e.g. ``...@enacted`` vs
   ``...@current``)?
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..retrieval.synthesize import Answer
from .schema import GoldQuestion


@dataclass(slots=True)
class QuestionScore:
    """All metric values for a single (question, answer) pair."""

    question_id: str
    task_type: str
    recall_at_k: float | None = None
    citation_accuracy: float | None = None
    good_law_precision: float | None = None
    good_law_recall: float | None = None
    good_law_named_treater: bool | None = None
    as_of_correct: bool | None = None
    fabricated_citations: list[str] = field(default_factory=list)
    missing_authorities: list[str] = field(default_factory=list)
    surfaced_authorities: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --- per-metric helpers ----------------------------------------------------


def _authorities_in_answer(answer: Answer) -> set[str]:
    """Identifiers the retrieval pipeline *surfaced* — citations + candidates.

    The brief asks "did the right authorities get retrieved?", which is wider
    than "did the model write them down" — a candidate that landed in the
    reranked context but didn't get cited still counts as retrieved. We use
    the union so a tight-mouthed model doesn't fail recall when retrieval
    actually nailed it.
    """
    out: set[str] = set()
    for c in answer.used_candidates:
        if c.parent_id:
            out.add(c.parent_id)
    out.update(answer.citations)
    return out


def recall_at_k(answer: Answer, expected: Iterable[str]) -> float | None:
    exp = set(expected)
    if not exp:
        return None
    surfaced = _authorities_in_answer(answer)
    return len(exp & surfaced) / len(exp)


def citation_accuracy(answer: Answer) -> float | None:
    """Fraction of cited ids that aren't fabricated. ``None`` if no citations."""
    if not answer.citations and not _fabricated_in_answer(answer):
        return None
    cited = answer.citations + _fabricated_in_answer(answer)
    if not cited:
        return None
    return len(answer.citations) / len(cited)


def _fabricated_in_answer(answer: Answer) -> list[str]:
    """Re-extract fabrication list from caveats so we don't lose info."""
    out: list[str] = []
    for cv in answer.caveats:
        if cv.startswith("WARNING — model emitted citations not present"):
            tail = cv.split(":", 1)[-1].strip()
            out.extend(x.strip() for x in tail.split(",") if x.strip())
    return out


def good_law_precision_recall(
    answer: Answer,
    question: GoldQuestion,
) -> tuple[float | None, float | None, bool | None]:
    """Return ``(precision, recall, named_treater)`` for a good-law question.

    Precision: of the cases the answer flagged as adverse-treatment, how
    many match the gold set?
    Recall: of the cases the gold says are adverse-treated, how many did
    the answer actually flag?
    Named treater: when ``expected_treating_case`` is set, did the answer
    text or caveats include that identifier verbatim?
    """
    expected_flagged = set(question.expected_good_law)
    if not expected_flagged:
        return None, None, None

    # Our caveat shape is "<TargetName> [<target_id>] — overruled by <TreaterName> [<treater_id>]".
    # We only treat the FIRST bracketed id per caveat as flagged (the target);
    # the second one is the treating case, which has a separate gold field.
    flagged_in_answer: set[str] = set()
    for cv in answer.caveats:
        idents = _extract_bracketed(cv)
        if idents:
            flagged_in_answer.add(idents[0])

    tp = len(expected_flagged & flagged_in_answer)
    prec = tp / len(flagged_in_answer) if flagged_in_answer else 0.0
    rec = tp / len(expected_flagged) if expected_flagged else 0.0

    named = None
    if question.expected_treating_case:
        text = answer.text + " " + " ".join(answer.caveats)
        named = question.expected_treating_case in text
    return prec, rec, named


def _extract_bracketed(text: str) -> list[str]:
    import re

    return re.findall(r"\[([^\[\]]+)\]", text)


def as_of_correct(answer: Answer, expected: Iterable[str]) -> bool | None:
    """Did the system surface *every* expected versioned Provision identifier?

    ``expected`` for an as-of-date question is the list of version-pinned
    Provision ids (e.g. ``uk/.../section/3@enacted``). We check that each
    appears in the retrieved/cited set — same as recall but stricter (every
    one must be present).
    """
    exp = list(expected)
    if not exp:
        return None
    surfaced = _authorities_in_answer(answer)
    return all(e in surfaced for e in exp)


# --- single-question driver -----------------------------------------------


def score_question(question: GoldQuestion, answer: Answer) -> QuestionScore:
    surfaced = _authorities_in_answer(answer)
    expected = set(question.expected_authorities)
    missing = sorted(expected - surfaced)
    notes: list[str] = []

    recall = recall_at_k(answer, expected)
    cite_acc = citation_accuracy(answer)
    fabricated = _fabricated_in_answer(answer)

    gl_prec, gl_rec, gl_named = good_law_precision_recall(answer, question)

    as_of_ok: bool | None = None
    if question.task_type == "as_of_date":
        as_of_ok = as_of_correct(answer, expected)

    # No-fabrication tasks invert the contract: missing authorities are fine,
    # fabricated ones are catastrophic.
    if question.task_type == "no_fabrication":
        notes.append("no_fabrication: must not invent a citation")
        if fabricated:
            cite_acc = 0.0

    return QuestionScore(
        question_id=question.id,
        task_type=question.task_type,
        recall_at_k=recall,
        citation_accuracy=cite_acc,
        good_law_precision=gl_prec,
        good_law_recall=gl_rec,
        good_law_named_treater=gl_named,
        as_of_correct=as_of_ok,
        fabricated_citations=fabricated,
        missing_authorities=missing,
        surfaced_authorities=sorted(surfaced),
        notes=notes,
    )


# --- aggregation ----------------------------------------------------------


def _mean(values: list[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None


@dataclass(slots=True)
class AggregateScores:
    """One-row summary across a list of ``QuestionScore``."""

    n: int
    n_with_recall: int
    recall_at_k_mean: float | None
    n_with_citation_accuracy: int
    citation_accuracy_mean: float | None
    citation_accuracy_perfect: float | None
    n_with_good_law: int
    good_law_precision_mean: float | None
    good_law_recall_mean: float | None
    good_law_named_treater_rate: float | None
    n_as_of: int
    as_of_correct_rate: float | None
    fabrication_rate: float
    questions_with_fabrication: int


def aggregate(scores: list[QuestionScore]) -> AggregateScores:
    if not scores:
        return AggregateScores(
            n=0,
            n_with_recall=0,
            recall_at_k_mean=None,
            n_with_citation_accuracy=0,
            citation_accuracy_mean=None,
            citation_accuracy_perfect=None,
            n_with_good_law=0,
            good_law_precision_mean=None,
            good_law_recall_mean=None,
            good_law_named_treater_rate=None,
            n_as_of=0,
            as_of_correct_rate=None,
            fabrication_rate=0.0,
            questions_with_fabrication=0,
        )
    n = len(scores)
    recalls = [s.recall_at_k for s in scores]
    cite_accs = [s.citation_accuracy for s in scores]
    gl_precs = [s.good_law_precision for s in scores]
    gl_recs = [s.good_law_recall for s in scores]
    named = [s.good_law_named_treater for s in scores if s.good_law_named_treater is not None]
    as_of_vals = [s.as_of_correct for s in scores if s.as_of_correct is not None]
    n_fab = sum(1 for s in scores if s.fabricated_citations)

    n_with_recall = sum(1 for v in recalls if v is not None)
    n_with_cite = sum(1 for v in cite_accs if v is not None)
    n_with_gl = sum(1 for v in gl_precs if v is not None)
    perfect_cite = (
        sum(1 for v in cite_accs if v is not None and v == 1.0) / n_with_cite
        if n_with_cite
        else None
    )
    return AggregateScores(
        n=n,
        n_with_recall=n_with_recall,
        recall_at_k_mean=_mean(recalls),
        n_with_citation_accuracy=n_with_cite,
        citation_accuracy_mean=_mean(cite_accs),
        citation_accuracy_perfect=perfect_cite,
        n_with_good_law=n_with_gl,
        good_law_precision_mean=_mean(gl_precs),
        good_law_recall_mean=_mean(gl_recs),
        good_law_named_treater_rate=(sum(1 for v in named if v) / len(named)) if named else None,
        n_as_of=len(as_of_vals),
        as_of_correct_rate=(sum(1 for v in as_of_vals if v) / len(as_of_vals))
        if as_of_vals
        else None,
        fabrication_rate=n_fab / n,
        questions_with_fabrication=n_fab,
    )
