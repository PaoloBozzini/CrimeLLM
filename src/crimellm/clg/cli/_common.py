"""Shared CLI helpers (Phase-stub placeholder, dest-dir defaults)."""

from __future__ import annotations

from pathlib import Path

from ..config import get_settings

PENDING = "Phase 0 stub — not implemented yet."


def cl_raw_dir(raw_dir: Path | None = None) -> Path:
    """Default the CourtListener raw-dir to ``data/raw/courtlistener/``."""
    return raw_dir or (get_settings().raw_root / "courtlistener")
