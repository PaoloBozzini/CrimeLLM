"""Exception types the autofetch worker treats specially.

``UnsupportedCite`` is the only non-failure signal a Source can raise to
short-circuit a job. Everything else propagates as ``FAILED`` (attempts
bump, breaker counts the failure).
"""

from __future__ import annotations


class UnsupportedCite(Exception):
    """Raised by ``Source.fetch_one`` when the cite shape is unfetchable.

    Examples:
    - ``DK/<short_title>/section/279`` — resolver maps it to retsinformation,
      but there is no slug → accn lookup table. Future work.
    - Karnov-gated Ufr citation with no subscription key configured.
    - ECLI:DK case law without a domstol scraper.

    The worker marks the queue row ``status='skipped'`` (terminal, with the
    exception message as the reason) and does NOT increment the breaker's
    failure counter — the source isn't broken, the cite is.
    """
