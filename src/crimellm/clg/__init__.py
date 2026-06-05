"""Common Legal Graph (clg) — Neo4j graph RAG pipeline over US + UK primary law.

Phase 0 milestone: Neo4j running via docker-compose, schema bootstrapped,
CLI exposes all pipeline subcommands. Subsequent phases populate the graph.
"""

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
