"""Deprecated shim. Re-exports the device helpers from ``crimellm.common.device``.

``device.py`` used to live in ``classifier/`` because it was first written for
the fine-tune pipeline. The CUDA/MPS/CPU detection is generic — the clg
distillation tier and any future ML stack want the same logic. The canonical
module is now ``crimellm.common.device``.

Scheduled for removal in v0.3.
"""

from __future__ import annotations

import warnings

from ..common.device import (  # noqa: F401
    DeviceInfo,
    resolve_device,
    training_kwargs_for_device,
)

warnings.warn(
    "crimellm.classifier.device is deprecated; import from "
    "crimellm.common.device (or crimellm.classifier — back-compat re-export) "
    "instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DeviceInfo", "resolve_device", "training_kwargs_for_device"]
