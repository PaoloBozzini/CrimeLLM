"""Swappable embedder for the lexical layer (Phase 3).

Backends:

* ``SentenceTransformerEmbedder`` — fully local via ``sentence-transformers``
  (which transparently loads any HF encoder). The **production default** is
  ``Qwen/Qwen3-Embedding-8B`` (4096-d, multilingual incl. Danish, MTEB
  top-tier, no API key). Fallback: ``BAAI/bge-m3`` (1024-d, fast on CPU).
  For dev / CI: ``Qwen/Qwen3-Embedding-0.6B`` (1024-d) or
  ``sentence-transformers/all-MiniLM-L6-v2`` (384-d).
* ``VoyageEmbedder`` — Voyage AI cloud (``voyage-multilingual-2``, 1024-d).
  Kept as an optional path; requires ``VOYAGE_API_KEY``.
* ``OpenAIEmbedder`` — ``text-embedding-3-large`` fallback. Requires
  ``OPENAI_API_KEY``.
* ``FakeEmbedder`` — deterministic hash-based embedder. Zero-cost, no
  network, perfect for tests. Outputs ``dim``-dimensional L2-normalised
  unit vectors derived from ``sha256(text)``.

All implement the same ``Embedder`` ABC. The factory ``get_embedder``
reads ``clg.config.Settings`` (model name, dim, API keys) and returns the
right backend. Model names with an ``org/model`` shape route to
sentence-transformers; explicit ``--backend voyage|openai|fake`` overrides.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..config import Settings, get_settings


@dataclass
class EmbedResult:
    """Container for a single embedding plus its source text + provenance."""

    text: str
    vector: list[float]
    model: str


class Embedder(ABC):
    """Sequence-aware text embedder. ``embed_batch`` is the workhorse."""

    name: str
    dim: int

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text, in order."""

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


# --- Voyage ----------------------------------------------------------------


class VoyageEmbedder(Embedder):
    name = "voyage-multilingual-2"

    def __init__(self, api_key: str, model: str = "voyage-multilingual-2", dim: int = 1024):
        try:
            import voyageai
        except ImportError as e:  # pragma: no cover — caller installs the extra
            raise ImportError("voyageai not installed. Add the [clg] extra.") from e

        self.dim = dim
        self.name = model
        self._client = voyageai.Client(api_key=api_key)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # Voyage's `embed` batches and returns `.embeddings`. ``input_type``
        # of "document" matches our use (passages get indexed; queries use
        # "query" — exposed via a separate path if/when we need it).
        result = self._client.embed(list(texts), model=self.name, input_type="document")
        return [list(v) for v in result.embeddings]


# --- OpenAI ---------------------------------------------------------------


class OpenAIEmbedder(Embedder):
    name = "text-embedding-3-large"

    def __init__(self, api_key: str, model: str = "text-embedding-3-large", dim: int = 3072):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("openai not installed; add it as an extra.") from e
        self.dim = dim
        self.name = model
        self._client = OpenAI(api_key=api_key)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI rejects empty strings. Replace with a single space to keep
        # caller indexing intact; the resulting vector is meaningless but the
        # caller usually drops empty passages upstream anyway.
        cleaned = [t or " " for t in texts]
        resp = self._client.embeddings.create(model=self.name, input=cleaned)
        return [list(d.embedding) for d in resp.data]


# --- Sentence-Transformers (fully local) ----------------------------------


class SentenceTransformerEmbedder(Embedder):
    """Local embedder via ``sentence-transformers``.

    Default model is ``Qwen/Qwen3-Embedding-8B`` (4096-d, multilingual,
    ~16 GB). Runs on CPU by default; pass ``device='cuda'`` or ``'mps'``
    for GPU. Reuses the ``[classifier]`` extra (``sentence-transformers``
    is already declared there); no clg-extra addition needed.

    Smaller picks:

    * ``BAAI/bge-m3`` — 1024-d, ~2 GB, strong multilingual baseline.
    * ``Qwen/Qwen3-Embedding-0.6B`` — 1024-d, ~1.2 GB, CPU-friendly dev.
    * ``sentence-transformers/all-MiniLM-L6-v2`` — 384-d, ~25 MB, tests.

    First call downloads + caches the model under ``~/.cache/huggingface/``;
    subsequent calls are network-free.
    """

    name = "Qwen/Qwen3-Embedding-8B"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        device: str | None = None,
        dim: int | None = None,
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover — caller installs the extra
            raise ImportError(
                "sentence-transformers not installed. Add the [classifier] extra "
                "(or `uv pip install sentence-transformers`)."
            ) from e

        self.name = model
        self._normalize = normalize
        self._model = SentenceTransformer(model, device=device)
        self.dim = int(dim or self._model.get_sentence_embedding_dimension())

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # convert_to_numpy keeps memory bounded for large batches; we cast
        # to plain lists at the boundary so the Neo4j driver is happy.
        arr = self._model.encode(
            list(texts),
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in arr]


# --- Fake (tests + offline dev) -------------------------------------------


class FakeEmbedder(Embedder):
    """Deterministic, network-free embedder.

    Hashes the input with SHA-256 then expands the hash into ``dim`` floats
    by repeated hashing. Output is L2-normalised so cosine similarity sees
    unit vectors. Same text -> same vector; different text -> diverging
    vectors. Useful for tests where we just need the vector index to round-
    trip + retrieve a known seed.
    """

    name = "fake"

    def __init__(self, dim: int = 16, model: str = "fake"):
        self.dim = dim
        self.name = model

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        out: list[float] = []
        block_n = 0
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
        while len(out) < self.dim:
            # 32 bytes per sha256; map each byte to a float in [-1, 1).
            for b in digest:
                out.append((b / 127.5) - 1.0)
                if len(out) >= self.dim:
                    break
            block_n += 1
            digest = hashlib.sha256(digest + block_n.to_bytes(4, "big")).digest()
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]


# --- Factory ---------------------------------------------------------------


# HF org prefixes that route to sentence-transformers without an explicit
# ``--backend st`` flag. Extend when adopting new ecosystems (jinaai/,
# intfloat/, etc.). Cloud model names (``voyage-*``, ``text-embedding-*``)
# don't contain ``/`` so they're unaffected.
_HF_PREFIXES: tuple[str, ...] = (
    "sentence-transformers/",
    "Qwen/",
    "BAAI/",
    "intfloat/",
    "jinaai/",
    "nomic-ai/",
    "google/",
    "microsoft/",
    "mixedbread-ai/",
)

# Short aliases for the two open-source production picks.
_MODEL_ALIASES: dict[str, str] = {
    "qwen": "Qwen/Qwen3-Embedding-8B",
    "qwen3": "Qwen/Qwen3-Embedding-8B",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-dev": "Qwen/Qwen3-Embedding-0.6B",
    "bge-m3": "BAAI/bge-m3",
    "bgem3": "BAAI/bge-m3",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def _is_hf_model_name(name: str) -> bool:
    return any(name.startswith(p) for p in _HF_PREFIXES)


def get_embedder(
    backend: str | None = None,
    *,
    settings: Settings | None = None,
    model: str | None = None,
    device: str | None = None,
) -> Embedder:
    """Pick a backend by name or by config.

    Recognised backends (case-insensitive, with aliases):

    * ``sentence-transformers`` / ``st`` / ``local`` / ``hf`` — fully-local
      via ``sentence-transformers``. Default model
      ``Qwen/Qwen3-Embedding-8B`` (4096-d). Short aliases also accepted as
      ``--backend qwen | qwen3-0.6b | bge-m3 | minilm``.
    * ``voyage`` — Voyage AI cloud, needs ``VOYAGE_API_KEY``.
    * ``openai`` — OpenAI embeddings, needs ``OPENAI_API_KEY``.
    * ``fake`` — deterministic SHA-256 stub for tests / offline.

    Resolution order when ``backend`` is None: if the configured
    ``embedding_model`` looks like an HF path (``org/model``) →
    sentence-transformers; else Voyage if its key is set; else OpenAI if
    its key is set; else ``fake``.
    """
    settings = settings or get_settings()
    chosen_model = model or settings.embedding_model

    # Convenience alias resolution — `--model qwen` works without typing the
    # full HF path. Settings-level overrides should already use canonical
    # names so this mainly serves the CLI ergonomics.
    if chosen_model in _MODEL_ALIASES:
        chosen_model = _MODEL_ALIASES[chosen_model]
    if backend in _MODEL_ALIASES:
        chosen_model = _MODEL_ALIASES[backend]
        backend = "sentence-transformers"

    if backend is None:
        if _is_hf_model_name(chosen_model):
            backend = "sentence-transformers"
        elif settings.voyage_api_key:
            backend = "voyage"
        else:
            import os

            backend = "openai" if os.environ.get("OPENAI_API_KEY") else "fake"

    backend = backend.lower()
    if backend == "voyage":
        if not settings.voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set; cannot use Voyage embedder.")
        return VoyageEmbedder(
            api_key=settings.voyage_api_key,
            model=chosen_model,
            dim=settings.embedding_dim,
        )
    if backend == "openai":
        import os

        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use OpenAI embedder.")
        return OpenAIEmbedder(api_key=key, model=chosen_model)
    if backend in {"sentence-transformers", "st", "local", "hf"}:
        return SentenceTransformerEmbedder(model=chosen_model, device=device)
    if backend == "fake":
        return FakeEmbedder(dim=settings.embedding_dim)
    raise ValueError(
        f"unknown embedder backend {backend!r}; "
        "pick sentence-transformers / voyage / openai / fake (or short aliases qwen / bge-m3 / minilm)."
    )


def embed_in_batches(
    embedder: Embedder, texts: Iterable[str], *, batch_size: int = 64
) -> list[list[float]]:
    """Convenience: shovel any iterable through ``embed_batch`` in chunks."""
    buf: list[str] = []
    out: list[list[float]] = []
    for t in texts:
        buf.append(t)
        if len(buf) >= batch_size:
            out.extend(embedder.embed_batch(buf))
            buf = []
    if buf:
        out.extend(embedder.embed_batch(buf))
    return out
