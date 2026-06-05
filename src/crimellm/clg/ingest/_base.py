"""Common ABC for ingest sources.

Each new source (CourtListener, legislation.gov.uk, Find Case Law, eCFR,
EUR-Lex …) implements ``Source`` and is dispatched through the CLI by name.
The ABC owns the lifecycle (download -> parse -> load) and the small bit of
shared state (``IngestContext``) so per-source modules stay focused on what
their data actually looks like.

CourtListener (`clg/ingest/courtlistener.py` + `clg/parse/courtlistener.py`)
landed before this ABC and reads as a procedural module today. Phase 2 will
adapt it to ``Source`` while wiring up the second source — that's the right
moment for the abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..graph.driver import Neo4jStore, get_store


@dataclass
class IngestContext:
    """Per-run state passed through the pipeline.

    ``params`` is a free-form bag for source-specific knobs (e.g. dump date,
    cluster limit) so the ABC's method signatures stay narrow.
    """

    settings: Settings = field(default_factory=get_settings)
    store: Neo4jStore = field(default_factory=get_store)
    raw_dir: Path | None = None
    interim_dir: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def source_raw_dir(self, source_name: str) -> Path:
        base = self.raw_dir or self.settings.raw_root
        out = base / source_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def source_interim_dir(self, source_name: str) -> Path:
        base = self.interim_dir or self.settings.interim_root
        out = base / source_name
        out.mkdir(parents=True, exist_ok=True)
        return out


@dataclass
class LoadReport:
    """Summary returned from ``Source.load(...)``."""

    source: str
    counts: dict[str, int] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class Source(ABC):
    """Lifecycle: ``download`` → ``parse`` → ``load``.

    - ``download`` is resumable, polite, idempotent. Writes to ``raw_dir``.
    - ``parse`` streams from ``raw_dir`` and yields ``clg.models`` instances
      (or whatever the loader expects). Memory-light.
    - ``load`` consumes the parser's output and MERGEs into Neo4j via
      ``clg.graph.loaders``. Returns a ``LoadReport``.

    Subclasses pick how granular their ``parse`` return type is — most will
    yield tagged tuples like ``("court", Court(...))`` so the loader can
    dispatch without holding everything in memory.
    """

    name: str

    @abstractmethod
    def download(self, ctx: IngestContext) -> dict[str, Path]:
        """Fetch source files. Return ``{logical_name: local_path}``."""

    @abstractmethod
    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:
        """Stream ``(kind, model)`` pairs from the downloaded files."""

    @abstractmethod
    def load(self, ctx: IngestContext) -> LoadReport:
        """Run parse + push to Neo4j; return counts."""
