"""Karnov Online — commercial DK reporter (skeleton, licence-gated).

Phase 5 ships a deliberately empty implementation. Karnov Online and the
Ugeskrift for Retsvæsen (Ufr) are commercial reporters covering the full
Højesteret + landsret + selected byret corpus with editorial summaries
and bidirectional treatment annotations. They're the canonical reporter
citations a DK lawyer reads (``U.2010.1234H`` style).

We don't build the real ingester until the firm confirms a subscription
and shares API credentials. Constructing ``KarnovSource`` without a key
raises a clear error so a misconfigured pipeline run fails fast rather
than silently writing nothing.

When the subscription lands, expand this module to:

* Authenticate against the Karnov API (token / cookie / IP-allowlist —
  depends on the contract terms).
* Pull the reporter JSON / XML feed and parse into ``Case`` + treatment
  annotations.
* Wire treatment hints into the Phase 6 cascade as a high-precision tier.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..ingest._base import IngestContext, LoadReport, Source


@dataclass
class KarnovSource(Source):
    """Skeleton — refuses to run without ``karnov_api_key`` in settings."""

    name: str = "karnov"
    items: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        key = get_settings().karnov_api_key
        if not key:
            raise RuntimeError(
                "Karnov ingester is a skeleton: KARNOV_API_KEY is not set. "
                "The firm must hold a Karnov Online subscription before this "
                "module can be wired up. Set KARNOV_API_KEY in .env once the "
                "credentials are issued."
            )

    def download(self, ctx: IngestContext) -> dict[str, Path]:  # pragma: no cover
        raise NotImplementedError(
            "Karnov ingester not implemented yet — see ingest/karnov.py docstring."
        )

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:  # pragma: no cover
        raise NotImplementedError(
            "Karnov ingester not implemented yet — see ingest/karnov.py docstring."
        )
        yield  # pragma: no cover - make this a generator for typing

    def load(self, ctx: IngestContext) -> LoadReport:  # pragma: no cover
        raise NotImplementedError(
            "Karnov ingester not implemented yet — see ingest/karnov.py docstring."
        )
