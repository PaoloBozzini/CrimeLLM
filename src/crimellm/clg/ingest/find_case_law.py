"""Find Case Law (TNA) — UK judgments downloader.

**Licence:** programmatic / bulk extraction from caselaw.nationalarchives.gov.uk
requires applying for the (free) computational-analysis licence. The Source
refuses to operate unless ``TNA_COMPUTATIONAL_LICENCE_ACCEPTED=1`` is set
(see ``crimellm.clg.config.Settings``). Rate limit: ≈1 000 requests / 5 min.

Bulk-feed wiring is intentionally minimal here; the Phase 2 gate only needs
the Source ABC + loader plumbing to be in place and a fixture-driven test
that proves the INTERPRETS join works. A first real-data run lands when the
licence is in hand.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ...common.http import UA, get_with_retry
from ..ingest._base import IngestContext, LoadReport, Source
from ..models import Case
from ..parse import find_case_law as P

FCL_BASE = "https://caselaw.nationalarchives.gov.uk"


@dataclass
class FindCaseLawSource(Source):
    name: str = "find_case_law"
    # Pre-fetched judgment URIs (Atom feed listing TBD). For now callers
    # either supply local XML files via raw_dir, or hand over an explicit
    # list of judgment uris.
    judgment_uris: tuple[str, ...] = field(default_factory=tuple)

    def _enforce_licence(self, ctx: IngestContext) -> None:
        if not ctx.settings.tna_computational_licence_accepted:
            raise PermissionError(
                "Find Case Law bulk extraction requires the computational-analysis "
                "licence. Apply at https://caselaw.nationalarchives.gov.uk/ and set "
                "TNA_COMPUTATIONAL_LICENCE_ACCEPTED=1 in your .env."
            )

    def download(self, ctx: IngestContext) -> dict[str, Path]:
        self._enforce_licence(ctx)
        dest = ctx.source_raw_dir(self.name)
        out: dict[str, Path] = {}
        if not self.judgment_uris:
            return out
        with httpx.Client(headers=UA, timeout=60.0) as client:
            for uri in self.judgment_uris:
                slug = uri.strip("/").replace("/", "_")
                fp = dest / f"{slug}.xml"
                if fp.exists():
                    out[uri] = fp
                    continue
                url = uri if uri.startswith("http") else f"{FCL_BASE}{uri}/data.xml"
                r = get_with_retry(client, url)
                fp.write_bytes(r.content)
                out[uri] = fp
        return out

    def iter_judgments(self, ctx: IngestContext) -> Iterator[tuple[Case, list[P.SectionRef]]]:
        """Yield ``(Case, [SectionRef])`` for every cached judgment XML."""
        dest = ctx.source_raw_dir(self.name)
        for xml_file in sorted(dest.glob("*.xml")):
            yield P.parse_judgment_file(xml_file)

    def parse(self, ctx: IngestContext) -> Iterator[tuple[str, Any]]:
        for case, refs in self.iter_judgments(ctx):
            yield ("case", case)
            for ref in refs:
                yield ("interprets", (case.id, case.decision_date, ref))

    def load(self, ctx: IngestContext) -> LoadReport:
        from ..graph.loaders import load_cases, load_interprets

        cases: list[Case] = []
        interprets: list[tuple[str, Any, P.SectionRef]] = []
        for kind, item in self.parse(ctx):
            if kind == "case":
                cases.append(item)
            elif kind == "interprets":
                interprets.append(item)

        n_cases = load_cases(cases, store=ctx.store)
        n_int = load_interprets(interprets, store=ctx.store)
        return LoadReport(
            source=self.name,
            counts={"cases": n_cases, "interprets": n_int},
            extras={"judgments": len(cases)},
        )
