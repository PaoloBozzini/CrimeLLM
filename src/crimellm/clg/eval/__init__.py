"""Gold-set harness + metrics. Phase 6."""

from .metrics import (
    AggregateScores,
    QuestionScore,
    aggregate,
    as_of_correct,
    citation_accuracy,
    good_law_precision_recall,
    recall_at_k,
    score_question,
)
from .report import to_dict, to_json, to_markdown
from .runner import EvalReport, run_eval
from .schema import GoldQuestion, GoldSet, load_gold_set

__all__ = [
    "GoldQuestion",
    "GoldSet",
    "load_gold_set",
    "QuestionScore",
    "AggregateScores",
    "score_question",
    "aggregate",
    "recall_at_k",
    "citation_accuracy",
    "good_law_precision_recall",
    "as_of_correct",
    "EvalReport",
    "run_eval",
    "to_dict",
    "to_json",
    "to_markdown",
]
