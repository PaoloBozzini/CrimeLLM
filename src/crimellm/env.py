"""Lightweight `.env` loader.

Use `load_env()` to load environment variables from a `.env` file at project
root (or any path you pass). Secrets like `COURTLISTENER_API_TOKEN` and
`ANTHROPIC_API_KEY` are then picked up via `os.environ.get(...)`.

Search order if `path=None`:
  1. `$CWD/.env`
  2. walk up parents until a `.env` is found, stopping at the filesystem root.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv as _dotenv_load


def find_dotenv(start: Path | str | None = None) -> Path | None:
    """Walk up from `start` (default: CWD) looking for a `.env` file."""
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        f = cand / ".env"
        if f.is_file():
            return f
    return None


def load_env(path: str | Path | None = None, override: bool = False) -> Path | None:
    """Load environment vars from a `.env` file. Returns the path used (or None).

    Args:
        path: explicit `.env` path. If None, walks up from CWD.
        override: if True, env values in the file replace existing process env.
    """
    if path is None:
        found = find_dotenv()
    else:
        found = Path(path)
        if not found.is_file():
            found = None
    if found is None:
        return None
    _dotenv_load(found, override=override)
    return found


def get_env(name: str, default: str | None = None) -> str | None:
    """Convenience: read an env var, returning `default` if unset or empty."""
    val = os.environ.get(name)
    return val if val else default
