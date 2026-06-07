"""Settings for the clg pipeline. All secrets via env (loaded from .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DEFAULT_ENABLED_JURISDICTIONS = ["US", "EW", "UK", "EU", "DK"]


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

    # Embeddings
    voyage_api_key: str | None = Field(default=None)
    embedding_model: str = Field(default="voyage-law-2")
    embedding_dim: int = Field(default=1024)  # voyage-law-2 = 1024
    embedding_fallback_model: str = Field(default="text-embedding-3-large")

    # LLM (extraction, treatment classification, synthesis)
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-4-7")

    # Source-data licences
    tna_computational_licence_accepted: bool = Field(default=False)

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
