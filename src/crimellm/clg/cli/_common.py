"""Shared CLI helpers (Phase-stub placeholder, dest-dir defaults)."""

from __future__ import annotations

from pathlib import Path

from ..config import get_settings

PENDING = "Phase 0 stub — not implemented yet."


def cl_raw_dir(raw_dir: Path | None = None) -> Path:
    """Default the CourtListener raw-dir to ``data/raw/courtlistener/``."""
    return raw_dir or (get_settings().raw_root / "courtlistener")


def parse_jurisdiction_csv(value: str) -> list[str]:
    """Split ``"dk,EU, us"`` → ``["DK", "EU", "US"]``. De-dupes, drops empty."""
    seen: set[str] = set()
    out: list[str] = []
    for part in value.split(","):
        code = part.strip().upper()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out
