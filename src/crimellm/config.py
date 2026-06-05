"""Deprecated shim. Re-exports the training ``Config`` from ``classifier.config``.

This module used to hold the fine-tune classifier ``Config`` dataclass. The
classifier stack now lives under ``crimellm.classifier``; import from there:

    from crimellm.classifier import Config       # preferred
    from crimellm.classifier.config import Config

Or use the top-level re-export:

    from crimellm import Config

Scheduled for removal in v0.3.
"""

from __future__ import annotations

import warnings

from .classifier.config import Config  # noqa: F401

warnings.warn(
    "crimellm.config is deprecated; import from crimellm.classifier (or "
    "crimellm.classifier.config) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Config"]
