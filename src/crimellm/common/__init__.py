"""Shared, cross-stack utilities.

Used by both the classifier/FAISS pipeline (`corpora.py`, `rag.py`, `train.py`)
and the clg graph pipeline (`crimellm.clg.*`). Keep this module dependency-light
so neither stack pays for the other.

* ``http``     — always available (httpx, pure-python stdlib).
* ``language`` — always available (pure stdlib). DA vs EN detector.
* ``device``   — requires the [classifier] extra (torch). Import lazily; the
  package init only re-exports device helpers when torch is reachable, so
  ``from crimellm.common.device import resolve_device`` keeps the same
  semantics whichever extra you installed.
"""

from .http import UA, get_with_retry, stream_download, write_jsonl
from .language import detect_language

__all__ = [
    "UA",
    "get_with_retry",
    "stream_download",
    "write_jsonl",
    "detect_language",
]

# Optional torch-dependent helpers. Only re-exported when torch is present;
# otherwise users must add the [classifier] extra and import explicitly.
try:
    from .device import DeviceInfo, resolve_device, training_kwargs_for_device

    __all__ += ["DeviceInfo", "resolve_device", "training_kwargs_for_device"]
except ImportError:  # pragma: no cover — torch absent from this install
    pass
