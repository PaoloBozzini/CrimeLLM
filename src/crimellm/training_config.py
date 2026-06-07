"""Deprecated shim. Re-exports ``Config`` from ``classifier.config``.

Brief stop on the way from ``crimellm.config`` (pre-Tier-A) to
``crimellm.classifier.config`` (Tier-D). Import from
``crimellm.classifier.config`` instead. Scheduled for removal in v0.3.
"""

from __future__ import annotations

import warnings

from .classifier.config import Config  # noqa: F401

warnings.warn(
    "crimellm.training_config is deprecated; import from crimellm.classifier.config instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Config"]
