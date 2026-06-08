"""Settings for the clg pipeline. All secrets via env (loaded from .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DEFAULT_ENABLED_JURISDICTIONS = ["US", "EW", "UK", "EU", "DK"]

# Embedding-model → output dimension. Used to auto-derive ``embedding_dim``
# when the user sets ``EMBEDDING_MODEL`` without a matching ``EMBEDDING_DIM``,
# which is the most common footgun when swapping models (the Neo4j vector
# index is dim-keyed, so a mismatch silently breaks retrieval).
#
# Add new models here as you adopt them. Unknown models fall back to whatever
# ``embedding_dim`` is set to (default 4096 to match Qwen3-Embedding-8B).
KNOWN_MODEL_DIMS: dict[str, int] = {
    # Qwen3-Embedding family — multilingual (119 langs incl. Danish),
    # MTEB top-tier. 8B is the production default; 0.6B is a CPU-friendly
    # dev pick that stays in the same 1024-d slot as BGE-M3.
    "Qwen/Qwen3-Embedding-8B": 4096,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    # BGE-M3 — best 1024-d open multilingual embedder, fast on CPU.
    "BAAI/bge-m3": 1024,
    # Sentence-Transformers shortlist for tests / very small dev runs.
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-roberta-large-v1": 1024,
    # Cloud fallbacks (kept for parity; the codebase no longer defaults to them).
    "voyage-multilingual-2": 1024,
    "voyage-law-2": 1024,
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}


def dim_for_model(model_name: str) -> int | None:
    """Return the known output dim for ``model_name``, or ``None``."""
    return KNOWN_MODEL_DIMS.get(model_name)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="crimellm-dev")
    neo4j_database: str = Field(default="neo4j")

    # Embeddings — open-source default. Qwen3-Embedding-8B is multilingual
    # (covers Danish), MTEB top-tier, no API key required. Self-hosted via
    # sentence-transformers / transformers; needs ~16 GB RAM and is
    # GPU-friendly. Drop to ``BAAI/bge-m3`` (1024-d) if 4096-d vectors are
    # too heavy, or ``Qwen/Qwen3-Embedding-0.6B`` (1024-d) for CPU dev.
    # ``embedding_dim`` auto-derives from ``KNOWN_MODEL_DIMS`` when not
    # explicitly set, so swapping ``EMBEDDING_MODEL`` doesn't silently
    # mismatch the Neo4j vector index.
    voyage_api_key: str | None = Field(default=None)
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-8B")
    embedding_dim: int = Field(default=4096)
    embedding_fallback_model: str = Field(default="BAAI/bge-m3")

    # LLM (extraction, treatment classification, synthesis)
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-4-7")

    # Source-data licences
    tna_computational_licence_accepted: bool = Field(default=False)

    # DK commercial reporter subscriptions (Karnov / Ufr). Both ingesters
    # are skeletons until the firm confirms a subscription; the keys gate
    # construction so a misconfigured run errors cleanly instead of
    # silently building nothing.
    karnov_api_key: str | None = Field(default=None)
    ufr_api_key: str | None = Field(default=None)

    # Data paths
    data_root: Path = Field(default=Path("data"))
    raw_root: Path = Field(default=Path("data/raw"))
    interim_root: Path = Field(default=Path("data/interim"))

    # Jurisdictions gated at ingest CLI + retrieval boundaries. Drop a code
    # here to retire a jurisdiction without deleting code/data. Env var
    # accepts CSV: ENABLED_JURISDICTIONS=US,EU,DK.
    enabled_jurisdictions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(_DEFAULT_ENABLED_JURISDICTIONS)
    )

    @field_validator("enabled_jurisdictions", mode="before")
    @classmethod
    def _parse_enabled_jurisdictions(cls, v: object) -> list[str]:
        if v is None or v == "":
            return list(_DEFAULT_ENABLED_JURISDICTIONS)
        if isinstance(v, str):
            parts = [s.strip().upper() for s in v.split(",")]
            return [p for p in parts if p]
        if isinstance(v, (list, tuple)):
            return [str(x).strip().upper() for x in v if str(x).strip()]
        raise TypeError(f"enabled_jurisdictions: unsupported type {type(v).__name__}")

    def is_enabled(self, jurisdiction: str) -> bool:
        return jurisdiction.upper() in {j.upper() for j in self.enabled_jurisdictions}

    # Autofetch (self-management): when a citation references a doc not in
    # Neo4j, the reconciliation worker fetches → parses → loads → embeds it
    # asynchronously. Gated off by default; see
    # ``docs/self-management-autofetch.local.md`` for the full design.
    # Source-QPS map caps per-source request rate so a cite-flood can't
    # hammer a third-party API; the worker reads it via the circuit breaker.
    autofetch_enabled: bool = Field(default=False)
    autofetch_queue_path: Path = Field(default=Path("data/autofetch.db"))
    autofetch_max_depth: int = Field(default=2)
    autofetch_max_attempts: int = Field(default=3)
    autofetch_circuit_open_seconds: int = Field(default=3600)
    autofetch_source_qps: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=lambda: {
            "eurlex": 0.5,
            "retsinformation": 1.0,
            "courtlistener": 2.0,
        }
    )

    @field_validator("autofetch_source_qps", mode="before")
    @classmethod
    def _parse_autofetch_source_qps(cls, v: object) -> dict[str, float]:
        """Accept env-CSV ``source:qps,source:qps`` or a dict.

        Pydantic env loading hands strings here; without this the dict field
        would raise. Empty / None falls back to the field's default factory.
        """
        if v is None or v == "":
            return {"eurlex": 0.5, "retsinformation": 1.0, "courtlistener": 2.0}
        if isinstance(v, dict):
            return {str(k): float(val) for k, val in v.items()}
        if isinstance(v, str):
            out: dict[str, float] = {}
            for pair in v.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                if ":" not in pair:
                    raise ValueError(
                        f"autofetch_source_qps: expected 'source:qps', got {pair!r}"
                    )
                k, rhs = pair.split(":", 1)
                out[k.strip()] = float(rhs.strip())
            return out
        raise TypeError(f"autofetch_source_qps: unsupported type {type(v).__name__}")

    @model_validator(mode="before")
    @classmethod
    def _derive_embedding_dim(cls, values: Any) -> Any:
        """Pull ``embedding_dim`` from ``KNOWN_MODEL_DIMS`` when the user set
        only ``embedding_model``.

        Without this, ``EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B`` in ``.env``
        with no ``EMBEDDING_DIM`` would silently use the default 4096 — fine
        for Qwen-8B, broken for any other model. The validator runs only
        when the dim is *not* explicitly set, so user overrides win.
        """
        if not isinstance(values, dict):
            return values
        # Pydantic v2 normalises field names already (case_sensitive=False).
        model_set = "embedding_model" in values and values["embedding_model"]
        dim_set = "embedding_dim" in values
        if model_set and not dim_set:
            derived = KNOWN_MODEL_DIMS.get(str(values["embedding_model"]))
            if derived is not None:
                values["embedding_dim"] = derived
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
