"""Parse the user's free-text question into a structured ``Query``.

Two knobs come out:

* ``jurisdiction`` — defaults to ``None`` (cross-jurisdiction). Heuristics
  bump it to ``US`` / ``UK`` / ``EW`` when the question makes it obvious
  ("U.S. Code §...", "Fraud Act 2006 s.2"). CLI ``--jurisdiction`` always
  wins over inference.
* ``as_of`` — defaults to today (UTC). An explicit ISO date anywhere in the
  prompt ("as of 2018-05-12") overrides. CLI ``--as-of`` always wins.

We *don't* extract entities here — that's the job of seed + expand. Parsing
stays small so it can be reasoned about + tested cheaply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from ..models import Jurisdiction

__all__ = ["Jurisdiction", "Query", "parse_query"]

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# Cue phrases that bias jurisdiction. Conservative: when in doubt, no bias.
_US_CUES = (
    "u.s.c",
    "usc",
    "us code",
    "u.s. code",
    "scotus",
    "federal court",
    "circuit court",
    "district court",
    "ninth circuit",
    "second circuit",
    "supreme court of the united states",
    "courtlistener",
)
_UK_CUES = (
    "uk",
    "united kingdom",
    "england",
    "wales",
    "ukpga",
    "ukla",
    "ewca",
    "ewhc",
    "uksc",
    "privy council",
    "legislation.gov.uk",
    "fraud act",
    "theft act",
    "bribery act",
    "modern slavery act",
)


@dataclass(slots=True)
class Query:
    """Structured user query."""

    raw: str
    jurisdiction: Jurisdiction | None
    as_of: date

    def with_overrides(
        self,
        *,
        jurisdiction: Jurisdiction | None = None,
        as_of: date | str | None = None,
    ) -> Query:
        new_as_of = self.as_of
        if as_of is not None:
            new_as_of = (
                as_of
                if isinstance(as_of, date)
                else datetime.strptime(as_of[:10], "%Y-%m-%d").date()
            )
        return Query(
            raw=self.raw,
            jurisdiction=jurisdiction if jurisdiction is not None else self.jurisdiction,
            as_of=new_as_of,
        )


def _infer_jurisdiction(text: str) -> Jurisdiction | None:
    lower = text.lower()
    us = sum(1 for cue in _US_CUES if cue in lower)
    uk = sum(1 for cue in _UK_CUES if cue in lower)
    if us > uk and us > 0:
        return "US"
    if uk > us and uk > 0:
        return "UK"
    return None


def _infer_as_of(text: str) -> date:
    m = _ISO_DATE_RE.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return date.today()


def parse_query(text: str) -> Query:
    """Heuristic parse. Cheap, deterministic, easy to override at the CLI."""
    return Query(
        raw=text.strip(),
        jurisdiction=_infer_jurisdiction(text),
        as_of=_infer_as_of(text),
    )
