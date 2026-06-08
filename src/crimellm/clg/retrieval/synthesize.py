"""Synthesise a grounded answer over the reranked candidate set.

Four backends — all behind the same ``Synthesizer`` ABC, all routed through
``_finalise_answer`` so the "no fabricated citations" contract is enforced
uniformly:

* ``AnthropicSynthesizer`` — Claude with prompt caching on the system block.
  Highest quality; needs ``ANTHROPIC_API_KEY``.
* ``OllamaSynthesizer`` — talks to a local ``ollama serve``; works fully
  offline with whatever model you've pulled (``qwen2.5:7b-instruct`` etc.).
  Zero per-query cost.
* ``AirLLMSynthesizer`` — local generation via AirLLM (layer-by-layer disk
  offload). Slowest, but runs huge models on commodity hardware; auto-uses
  MLX on Apple Silicon.
* ``FakeSynthesizer`` — deterministic stub. Quotes the top candidate by id;
  by design cannot fabricate. Used in gate tests.

Each returns an ``Answer`` with the raw text plus structured citation +
caveat lists so the caller can audit (and so we can guard against
fabricated citations programmatically).
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from .good_law import GoodLawFlag, summary_label
from .parse_query import Query
from .prompts import (
    DISCLAIMER_EN,
    disclaimer_for,
    format_candidates_block,
    no_context_message_for,
    system_prompt_for,
)
from .seed import Candidate

# Back-compat alias — the EN disclaimer is still the most-common default
# and external code that imported ``DISCLAIMER`` from this module shouldn't
# break. New callers should use ``disclaimer_for(query.language)``.
DISCLAIMER = DISCLAIMER_EN


@dataclass(slots=True)
class Answer:
    """What the synthesizer produced."""

    question: str
    text: str
    citations: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    used_candidates: list[Candidate] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "text": self.text,
            "citations": list(self.citations),
            "caveats": list(self.caveats),
            "used": [
                {
                    "parent_type": c.parent_type,
                    "parent_id": c.parent_id,
                    "parent_name": c.parent_name,
                    "score": round(c.score, 4),
                    "sources": sorted(set(c.extras.get("sources", []) or [c.source])),
                }
                for c in self.used_candidates
            ],
            "model": self.model,
        }


# --- helpers shared by both backends ---------------------------------------


def _context_block(candidates: Sequence[Candidate], *, language: str = "en") -> str:
    """Serialise the reranked candidates for the prompt.

    Identifiers in brackets are the ONLY citations the model is allowed to
    emit. The synthesizer prompt makes that rule explicit; the caller
    enforces it after the fact via ``check_citations``.

    Headings are enriched per jurisdiction via
    ``prompts.format_human_citation`` so DK lbk ids render as
    ``Aftaleloven § 36`` and EU CELEX renders as ``Reg (EU) 2016/679,
    Art. 6`` — without changing the bracketed canonical id.
    """
    return format_candidates_block(candidates, language=language)


def _allowed_identifiers(candidates: Sequence[Candidate]) -> set[str]:
    return {c.parent_id for c in candidates if c.parent_id}


_IDENT_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def extract_citations(text: str) -> list[str]:
    """Pull bracketed identifiers out of synthesizer output, dedup-stable."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _IDENT_PATTERN.finditer(text):
        cand = m.group(1).strip()
        if cand and cand not in seen:
            out.append(cand)
            seen.add(cand)
    return out


def check_citations(text: str, allowed: set[str]) -> tuple[list[str], list[str]]:
    """Return ``(valid, fabricated)`` citation lists from the answer text."""
    valid: list[str] = []
    fabricated: list[str] = []
    for ident in extract_citations(text):
        (valid if ident in allowed else fabricated).append(ident)
    return valid, fabricated


def caveats_from_good_law(
    candidates: Sequence[Candidate],
    flags: dict[str, list[GoodLawFlag]],
) -> list[str]:
    """One human caveat string per Case carrying adverse treatment."""
    out: list[str] = []
    for c in candidates:
        ff = flags.get(c.parent_id)
        if not ff:
            continue
        label = summary_label(ff)
        if label:
            out.append(f"{c.parent_name} [{c.parent_id}] — {label}.")
    return out


# --- Shared prompt + Answer assembly --------------------------------------


def _empty_answer(query: Query, model: str) -> Answer:
    """Returned when the retrieval pipeline handed us nothing to ground on."""
    return Answer(
        question=query.raw,
        text=(
            disclaimer_for(query.language) + "\n\n" + no_context_message_for(query.language)
        ),
        citations=[],
        caveats=[],
        used_candidates=[],
        model=model,
    )


_USER_PROMPT_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "question": "Question",
        "jurisdiction": "Jurisdiction",
        "as_of": "As-of date",
        "unspecified": "unspecified",
        "caveats_header": "Caveats (must surface verbatim)",
        "none": "(none)",
        "context": "Context",
    },
    "da": {
        "question": "Spørgsmål",
        "jurisdiction": "Jurisdiktion",
        "as_of": "As-of dato",
        "unspecified": "uspecificeret",
        "caveats_header": "Forbehold (skal gengives ordret)",
        "none": "(ingen)",
        "context": "Context",
    },
}


def _build_user_prompt(
    query: Query,
    candidates: Sequence[Candidate],
    caveats: list[str],
) -> str:
    """Standard user-message body shared by every LLM-backed synthesizer.

    Labels are localised to ``query.language``; ``"Context"`` stays in EN
    so the system prompt's strict-rules reference (which always names the
    ``Context`` block) matches regardless of caller language.
    """
    labels = _USER_PROMPT_LABELS.get(
        (query.language or "en").lower(), _USER_PROMPT_LABELS["en"]
    )
    return (
        f"{labels['question']}: {query.raw}\n"
        f"{labels['jurisdiction']}: {query.jurisdiction or labels['unspecified']}\n"
        f"{labels['as_of']}: {query.as_of.isoformat()}\n\n"
        f"{labels['caveats_header']}:\n"
        + (
            "\n".join(f"- {x}" for x in caveats)
            if caveats
            else f"- {labels['none']}"
        )
        + f"\n\n{labels['context']}:\n{_context_block(candidates, language=query.language)}"
    )


_FABRICATION_NOTE: dict[str, str] = {
    "en": "WARNING — model emitted citations not present in retrieved context: ",
    "da": "ADVARSEL — modellen genererede citater, som ikke findes i den hentede kontekst: ",
}


def _finalise_answer(
    *,
    query: Query,
    candidates: Sequence[Candidate],
    caveats: list[str],
    model_text: str,
    model_name: str,
) -> Answer:
    """Run the citation guard, build the ``Answer``, prepend the disclaimer.

    All LLM backends route through here so the "no fabricated citations"
    contract is enforced uniformly: extract every ``[id]`` from the model's
    output, split into valid vs not-in-context, and surface fabrications as a
    visible caveat. Disclaimer + fabrication note are language-routed.
    """
    allowed = _allowed_identifiers(candidates)
    valid, fabricated = check_citations(model_text, allowed)
    lang = (query.language or "en").lower()
    if fabricated:
        prefix = _FABRICATION_NOTE.get(lang, _FABRICATION_NOTE["en"])
        caveats.append(prefix + ", ".join(fabricated))
    body = (model_text or "").strip()
    return Answer(
        question=query.raw,
        text=disclaimer_for(lang) + "\n\n" + body,
        citations=valid,
        caveats=caveats,
        used_candidates=list(candidates),
        model=model_name,
    )


# --- ABC -------------------------------------------------------------------


class Synthesizer(ABC):
    name: str

    @abstractmethod
    def synthesise(
        self,
        *,
        query: Query,
        candidates: Sequence[Candidate],
        good_law: dict[str, list[GoodLawFlag]],
    ) -> Answer: ...


# --- Anthropic Claude ------------------------------------------------------


# Back-compat: external callers that imported ``_SYSTEM_PROMPT`` from this
# module still get the EN common-law prompt. Active routing uses
# ``system_prompt_for(query.language)`` per call.
from .prompts import SYSTEM_PROMPT_EN as _SYSTEM_PROMPT  # noqa: E402, F401


class AnthropicSynthesizer(Synthesizer):
    """Claude via the Anthropic SDK. Caches the system block.

    Lazy SDK import keeps the Phase 4 retrieval modules importable without
    the anthropic extra. The synthesizer fails at construction time with a
    clear message when the SDK or key is missing.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        max_tokens: int = 1024,
        api_key: str | None = None,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover — caller installs the extra
            raise ImportError(
                "anthropic package not installed. Add the [anthropic] or [clg] extra."
            ) from e

        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot use AnthropicSynthesizer.")

        self.model = model
        self._max_tokens = max_tokens
        self._client = Anthropic(api_key=key)

    def synthesise(
        self,
        *,
        query: Query,
        candidates: Sequence[Candidate],
        good_law: dict[str, list[GoodLawFlag]],
    ) -> Answer:
        if not candidates:
            return _empty_answer(query, self.model)

        caveats = caveats_from_good_law(candidates, good_law)
        user_prompt = _build_user_prompt(query, candidates, caveats)

        # Prompt caching: each language is its own cache entry, but
        # the per-language text is stable so subsequent calls in the same
        # language hit cache.
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt_for(query.language),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(getattr(p, "text", "") for p in msg.content) or ""
        return _finalise_answer(
            query=query,
            candidates=candidates,
            caveats=caveats,
            model_text=text,
            model_name=self.model,
        )


# --- Ollama (local; HTTP) -------------------------------------------------


class OllamaSynthesizer(Synthesizer):
    """Synthesise via a local Ollama server (no API key, no cloud).

    Same strict prompt as the Anthropic backend; output is passed through
    the same citation guard. Talks to ``http://localhost:11434/api/chat``
    by default — override with ``host``.

    Prereqs (once)::

        # macOS:    brew install ollama && ollama serve &
        # Linux:    https://ollama.com/download
        ollama pull qwen2.5:7b-instruct        # ~4.5 GB, fast + capable
        # or:  ollama pull llama3.1:8b-instruct   # ~5 GB
        # or:  ollama pull qwen2.5:14b-instruct   # ~9 GB, higher quality
    """

    name = "ollama"

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def synthesise(
        self,
        *,
        query: Query,
        candidates: Sequence[Candidate],
        good_law: dict[str, list[GoodLawFlag]],
    ) -> Answer:
        if not candidates:
            return _empty_answer(query, self.model)

        import httpx

        caveats = caveats_from_good_law(candidates, good_law)
        user_prompt = _build_user_prompt(query, candidates, caveats)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt_for(query.language)},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self._max_tokens,
            },
        }
        try:
            r = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self._timeout)
            r.raise_for_status()
            text = r.json().get("message", {}).get("content", "") or ""
        except httpx.HTTPError as e:
            return Answer(
                question=query.raw,
                text=disclaimer_for(query.language) + f"\n\nOllama call failed: {e!s}",
                citations=[],
                caveats=caveats + [f"Ollama backend error: {type(e).__name__}: {e}"],
                used_candidates=list(candidates),
                model=self.model,
            )
        return _finalise_answer(
            query=query,
            candidates=candidates,
            caveats=caveats,
            model_text=text,
            model_name=self.model,
        )


# --- AirLLM (local; layer-by-layer disk offload) --------------------------


class AirLLMSynthesizer(Synthesizer):
    """Synthesise via AirLLM (local generation with disk offload).

    Loads exactly one transformer layer at a time, so it runs huge models
    (Qwen2.5-7B, Llama-3.1-8B) on commodity hardware at the cost of speed.
    Auto-uses the MLX backend on Apple Silicon, CUDA elsewhere.

    Prereqs (once)::

        uv sync --extra clg --extra airllm           # any platform
        uv sync --extra clg --extra airllm-mlx       # Apple Silicon (faster)
        uv sync --extra clg --extra airllm-cuda      # NVIDIA + bitsandbytes
    """

    name = "airllm"

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        compression: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device: str | None = None,
    ):
        try:
            import airllm  # noqa: F401
        except ImportError as e:  # pragma: no cover — caller installs the extra
            raise ImportError(
                "airllm not installed. Add one of the [airllm], [airllm-mlx], "
                "or [airllm-cuda] extras."
            ) from e

        from airllm import AutoModel

        self.model = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        kwargs: dict[str, object] = {}
        if compression:
            kwargs["compression"] = compression
        if device:
            kwargs["device"] = device
        self._model = AutoModel.from_pretrained(model_id, **kwargs)

    def synthesise(
        self,
        *,
        query: Query,
        candidates: Sequence[Candidate],
        good_law: dict[str, list[GoodLawFlag]],
    ) -> Answer:
        if not candidates:
            return _empty_answer(query, self.model)

        caveats = caveats_from_good_law(candidates, good_law)
        user_prompt = _build_user_prompt(query, candidates, caveats)
        prompt = f"{system_prompt_for(query.language)}\n\n{user_prompt}"

        try:
            input_tokens = self._model.tokenizer(
                prompt, return_tensors="pt", return_attention_mask=False
            )
            output = self._model.generate(
                input_tokens["input_ids"],
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                use_cache=True,
                return_dict_in_generate=True,
            )
            ids = getattr(output, "sequences", output)
            try:
                seq = ids[0]
            except Exception:  # pragma: no cover — best-effort across backends
                seq = ids
            text = self._model.tokenizer.decode(seq, skip_special_tokens=True)
            # AirLLM echoes the prompt — strip it from the start.
            if text.startswith(prompt):
                text = text[len(prompt) :].lstrip()
        except Exception as e:  # noqa: BLE001
            return Answer(
                question=query.raw,
                text=disclaimer_for(query.language) + f"\n\nAirLLM call failed: {e!s}",
                citations=[],
                caveats=caveats + [f"AirLLM backend error: {type(e).__name__}: {e}"],
                used_candidates=list(candidates),
                model=self.model,
            )

        return _finalise_answer(
            query=query,
            candidates=candidates,
            caveats=caveats,
            model_text=text,
            model_name=self.model,
        )


# --- Fake (tests + offline dev) -------------------------------------------


class FakeSynthesizer(Synthesizer):
    """Deterministic answer: quotes the top candidate by its identifier.

    Always cites only identifiers from the retrieved context — by design it
    can never fabricate a citation, which makes it ideal for gate tests.
    """

    name = "fake"
    model = "fake"

    def synthesise(
        self,
        *,
        query: Query,
        candidates: Sequence[Candidate],
        good_law: dict[str, list[GoodLawFlag]],
    ) -> Answer:
        caveats = caveats_from_good_law(candidates, good_law)
        if not candidates:
            return _empty_answer(query, self.model)
        top = candidates[0]
        snippet = (top.text or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:500].rstrip() + "…"
        cited = sorted({c.parent_id for c in candidates if c.parent_id})
        lang = (query.language or "en").lower()
        if lang == "da":
            body = (
                f"Baseret på de hentede kilder er den nærmeste match "
                f"[{top.parent_id}]. Relevant tekst: {snippet}\n\n"
                f"Alle betragtede kilder: {', '.join(f'[{x}]' for x in cited)}."
            )
        else:
            body = (
                f"Based on the retrieved authorities, the closest match is "
                f"[{top.parent_id}]. Relevant text: {snippet}\n\n"
                f"All considered authorities: {', '.join(f'[{x}]' for x in cited)}."
            )
        return Answer(
            question=query.raw,
            text=disclaimer_for(lang) + "\n\n" + body,
            citations=cited,
            caveats=caveats,
            used_candidates=list(candidates),
            model=self.model,
        )


def _ollama_reachable(host: str = "http://localhost:11434", timeout: float = 0.5) -> bool:
    """Probe Ollama for liveness — used by the auto-pick path."""
    try:
        import httpx

        r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def get_synthesizer(
    name: str | None = None,
    *,
    model: str | None = None,
) -> Synthesizer:
    """Pick a synthesizer by name. Aliases supported.

    Recognised backends:

    * ``anthropic`` / ``claude`` — Claude via the SDK (prompt caching).
    * ``ollama`` — local model via a running Ollama server.
    * ``airllm`` — local generation via AirLLM (CPU/MLX/CUDA).
    * ``fake`` — deterministic stub, no network.

    Auto-pick (``name=None``) order: ``anthropic`` if ``ANTHROPIC_API_KEY``
    is set, else ``ollama`` if its server is reachable on localhost, else
    ``fake``. AirLLM must be requested explicitly — it does heavy load
    on first construction (model download + disk shard), which we don't
    want to do silently.
    """
    if name is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif _ollama_reachable():
            name = "ollama"
        else:
            name = "fake"
    n = name.lower()
    if n in {"anthropic", "claude"}:
        kwargs: dict[str, object] = {}
        if model:
            kwargs["model"] = model
        return AnthropicSynthesizer(**kwargs)
    if n in {"ollama", "local-llm"}:
        return OllamaSynthesizer(model=model or "qwen2.5:7b-instruct")
    if n == "airllm":
        return AirLLMSynthesizer(model_id=model or "Qwen/Qwen2.5-7B-Instruct")
    if n in {"fake", "test"}:
        return FakeSynthesizer()
    raise ValueError(f"unknown synthesizer {name!r}; pick anthropic / ollama / airllm / fake")
