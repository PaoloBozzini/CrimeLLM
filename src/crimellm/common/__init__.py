"""Shared, cross-stack utilities.

Used by both the classifier/FAISS pipeline (`corpora.py`, `rag.py`, `train.py`)
and the clg graph pipeline (`crimellm.clg.*`). Keep this module dependency-light
so neither stack pays for the other.
"""

from .http import UA, get_with_retry, stream_download, write_jsonl

__all__ = ["UA", "get_with_retry", "stream_download", "write_jsonl"]
