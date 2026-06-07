"""Source downloaders. Each is resumable, rate-limited, cached, provenance-tagged.

Phase 1 onwards. New sources implement ``Source`` from ``_base``; the
existing ``courtlistener`` module is a procedural sketch from Phase 1 and
will be adapted to the ABC alongside Phase 2's second source.
"""

from ._base import IngestContext, LoadReport, Source

__all__ = ["IngestContext", "LoadReport", "Source"]
