"""High-level orchestrator. ``run_query`` is what the CLI calls.

Flow: parse → seed (vector search) → expand (graph traversal, filtered by
jurisdiction + as-of) → good-law check → rerank → synthesise.
"""

from __future__ import annotations

from datetime import date as _date

from ..embed.embedder import Embedder, get_embedder
from ..graph.driver import Neo4jStore, get_store
from .expand import expand_seeds
from .good_law import check_good_law
from .parse_query import Jurisdiction, Query, parse_query
from .rerank import RerankWeights, rerank
from .seed import seed_from_chunks
from .synthesize import Answer, Synthesizer, get_synthesizer


def run_query(
    question: str,
    *,
    jurisdiction: Jurisdiction | None = None,
    as_of: str | _date | None = None,
    language: str | None = None,
    seed_k: int = 8,
    top_k: int = 6,
    weights: RerankWeights | None = None,
    embedder: Embedder | None = None,
    synthesizer: Synthesizer | None = None,
    embedder_backend: str | None = None,
    synthesizer_name: str | None = None,
    store: Neo4jStore | None = None,
) -> Answer:
    """End-to-end retrieval + synthesis.

    Sensible defaults for every knob — pass keyword args to override.
    ``embedder_backend`` / ``synthesizer_name`` are convenience hooks so the
    CLI doesn't have to construct the backends itself; they're ignored when
    explicit ``embedder`` / ``synthesizer`` are passed.

    ``language`` (``"en"`` / ``"da"``) overrides the detector when the
    caller wants to force a synthesis language regardless of what the
    question itself looks like — useful when an EN-speaking lawyer asks
    about a Danish statute and wants the answer in DA, or vice versa.
    """
    store = store or get_store()
    embedder = embedder or get_embedder(embedder_backend)
    synthesizer = synthesizer or get_synthesizer(synthesizer_name)

    query: Query = parse_query(question).with_overrides(
        jurisdiction=jurisdiction, as_of=as_of, language=language
    )

    # Autofetch: enqueue any canonical cite the question names that's not in
    # the graph yet. No-op when ``autofetch_enabled`` is false. The live
    # answer doesn't wait — the worker drains in the background.
    from ..autofetch.integration import enqueue_missing_for_query

    pending = enqueue_missing_for_query(query, store=store)

    seeds = seed_from_chunks(query, embedder, k=seed_k, store=store)
    expansions = expand_seeds(seeds, query=query, store=store)
    pooled = seeds + expansions

    case_ids = [c.parent_id for c in pooled if c.parent_type == "Case"]
    flags = check_good_law(case_ids, store=store)

    ranked = rerank(pooled, weights=weights, today=query.as_of, good_law=flags, top_k=top_k)
    answer = synthesizer.synthesise(query=query, candidates=ranked, good_law=flags)
    if pending:
        answer.pending_citations = list(pending)
    return answer
