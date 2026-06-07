"""Swappable embedder + chunker. Phase 3."""

from .chunker import chunk_case, chunk_provision, iter_chunks
from .embedder import (
    Embedder,
    FakeEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    VoyageEmbedder,
    embed_in_batches,
    get_embedder,
)

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "VoyageEmbedder",
    "get_embedder",
    "embed_in_batches",
    "chunk_case",
    "chunk_provision",
    "iter_chunks",
]
