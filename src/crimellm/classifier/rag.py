"""Classifier-stack FAISS retriever (``LegalRetriever``).

This is the original, flat-vector RAG retriever paired with the fine-tune
classifier. It is **not** the clg graph-RAG retriever (Phase 4 — that lives
under ``crimellm.clg.retrieval``). Both retrievers can coexist; pick based on
the task:

* ``LegalRetriever`` — quick lookup of statute/judgment snippets by
  similarity, single embedding model, FAISS IndexFlatIP. Good for the
  classifier and notebook demos.
* ``crimellm.clg.retrieval`` — graph traversal over the citation/treatment
  network with point-in-time + jurisdiction filters. Good for multi-hop +
  good-law + as-of-date queries.

Encodes a corpus of legal documents (statutes, judgments) with a
sentence-transformers model, indexes them in FAISS (cosine via inner-product
on L2-normalised vectors), and retrieves top-k relevant snippets for a query.

Corpus record schema (one JSON per line in ``<base>.jsonl``):

    {
        "id":       str,
        "text":     str,            # the chunk to embed + retrieve
        "source":   str,            # "us_code" | "courtlistener" | ...
        "citation": str,            # "18 U.S.C. § 1111" / "Brown v. Board ..."
        "type":     "statute" | "judgment",
        "metadata": dict,           # free-form extras
    }
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .embed_probe import encode_texts


@dataclass
class RetrievalHit:
    text: str
    source: str
    citation: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class LegalRetriever:
    """FAISS-backed dense retriever over a JSONL legal corpus.

    Persistence layout (given `base_path = "data/corpora/usc_title18"`):
        data/corpora/usc_title18.faiss      — FAISS IndexFlatIP
        data/corpora/usc_title18.jsonl      — one JSON record per line, same order as index
        data/corpora/usc_title18.meta.json  — embedding model name + dim
    """

    def __init__(
        self,
        index,
        records: list[dict[str, Any]],
        embedding_model_name: str,
        top_k: int = 5,
        _model=None,
    ):
        self._index = index
        self._records = records
        self._model_name = embedding_model_name
        self._model = _model  # may be None → lazy-load on first query
        self.top_k = top_k

    @classmethod
    def build(
        cls,
        documents: Sequence[dict[str, Any]],
        base_path: str | Path,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        top_k: int = 5,
    ) -> LegalRetriever:
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss-cpu not installed. Run: uv add faiss-cpu") from e

        texts = [d["text"] for d in documents]
        vecs, model = encode_texts(embedding_model, texts, batch_size=batch_size, normalize=True)
        vecs = np.ascontiguousarray(vecs.astype(np.float32))
        dim = int(vecs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(base_path) + ".faiss")
        with open(str(base_path) + ".jsonl", "w", encoding="utf-8") as f:
            for d in documents:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(str(base_path) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"embedding_model": embedding_model, "dim": dim}, f)

        return cls(index, list(documents), embedding_model, top_k=top_k, _model=model)

    @classmethod
    def load(cls, base_path: str | Path, top_k: int = 5) -> LegalRetriever:
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss-cpu not installed. Run: uv add faiss-cpu") from e

        base_path = Path(base_path)
        index = faiss.read_index(str(base_path) + ".faiss")
        records: list[dict[str, Any]] = []
        with open(str(base_path) + ".jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        with open(str(base_path) + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        if len(records) != index.ntotal:
            raise ValueError(
                f"index/records size mismatch: index={index.ntotal} jsonl={len(records)}"
            )
        return cls(index, records, meta["embedding_model"], top_k=top_k)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from .device import resolve_device

            backend = resolve_device().backend
            st_device = backend if backend in {"cuda", "mps", "cpu"} else "cpu"
            self._model = SentenceTransformer(self._model_name, device=st_device)
        k = k if k is not None else self.top_k
        qvec = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(
            np.float32
        )
        qvec = np.ascontiguousarray(qvec)
        scores, idxs = self._index.search(qvec, k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist(), strict=True):
            if idx < 0:
                continue
            rec = self._records[idx]
            hits.append(
                RetrievalHit(
                    text=rec["text"],
                    source=rec.get("source", ""),
                    citation=rec.get("citation", ""),
                    score=float(score),
                    metadata=rec.get("metadata", {}),
                )
            )
        return hits

    def __len__(self) -> int:
        return len(self._records)
