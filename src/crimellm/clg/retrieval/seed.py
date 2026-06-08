"""Seed candidates via vector search over ``Chunk``.

Wraps the existing ``graph.loaders.search_chunks`` helper and packages each
hit into a uniform ``Candidate`` shape that the expansion + rerank stages
consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..embed.embedder import Embedder
from ..graph.driver import Neo4jStore
from ..graph.loaders import search_chunks
from .parse_query import Query


@dataclass(slots=True)
class Candidate:
    """Single retrieved passage with full provenance back to its entity.

    ``source`` tracks where the candidate came from in the retrieval pipeline
    ("seed" / "cited_by" / "cites" / "interprets" / "as_of"). ``base_score``
    is what we have before any reranking — the raw vector cosine for seeds,
    a constant for traversal hops. The reranker writes a final ``score``.
    """

    chunk_id: str | None
    text: str
    parent_type: str
    parent_id: str
    parent_name: str
    parent_jurisdiction: str | None
    section_path: str | None = None
    version_id: str | None = None
    decision_date: Any = None  # str / date — driver may return either
    source: str = "seed"
    base_score: float = 0.0
    score: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def short_label(self) -> str:
        """Human-readable identifier used in answer citations + dedup keys."""
        if self.parent_type == "Provision":
            return f"{self.parent_id}"
        if self.parent_type == "Case" and self.parent_name:
            return f"{self.parent_name} [{self.parent_id}]"
        return self.parent_id


def seed_from_chunks(
    query: Query,
    embedder: Embedder,
    *,
    k: int = 8,
    store: Neo4jStore | None = None,
    enabled_jurisdictions: list[str] | None = None,
) -> list[Candidate]:
    """Run vector search → ``Candidate`` list, filtered by jurisdiction.

    When ``query.jurisdiction`` is set (CLI override or strong inference),
    the search is scoped to that one jurisdiction. Otherwise the search
    is scoped to ``enabled_jurisdictions`` (defaults to
    ``Settings.enabled_jurisdictions``). Pass ``["ALL"]`` or an empty list
    to disable the enabled-set filter — useful for admin/debug paths.
    """
    qvec = embedder.embed(query.raw)
    if enabled_jurisdictions is None:
        from ..config import get_settings

        enabled_jurisdictions = list(get_settings().enabled_jurisdictions)
    # Sentinel: empty list / ["ALL"] / ["*"] → disable enabled filter.
    if not enabled_jurisdictions or any(
        c.strip().upper() in {"ALL", "*"} for c in enabled_jurisdictions
    ):
        enabled_for_search: list[str] | None = None
    else:
        enabled_for_search = enabled_jurisdictions

    rows = search_chunks(
        qvec,
        k=k,
        jurisdiction=query.jurisdiction,
        enabled_jurisdictions=enabled_for_search,
        store=store,
    )
    out: list[Candidate] = []
    for r in rows:
        out.append(
            Candidate(
                chunk_id=r["chunk_id"],
                text=r["text"],
                parent_type=r["parent_type"],
                parent_id=r["parent_id"],
                parent_name=r["parent_name"] or r["parent_id"],
                parent_jurisdiction=r["parent_jurisdiction"],
                section_path=r["section_path"],
                version_id=r["version_id"],
                decision_date=r["decision_date"],
                source="seed",
                base_score=float(r["score"]),
            )
        )
    return out
