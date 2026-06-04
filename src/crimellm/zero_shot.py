"""Zero-shot LLM classification — no training, no labels.

Two backends:
  - OllamaClassifier   : local LLM via Ollama HTTP API (free, private, runs on
                         your Mac MPS or Windows CPU/GPU). Default backend.
  - AnthropicClassifier: Claude API (best quality, costs $, needs internet +
                         ANTHROPIC_API_KEY).

Both speak the same `classify(text) -> ZeroShotResult` interface.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable

import requests

DEFAULT_LABELS = ("no", "yes", "unclear")


def build_output_schema(labels: Iterable[str] = DEFAULT_LABELS) -> dict:
    """Single source of truth for the structured-output schema.

    Used by:
      - Ollama   : passed as `format=<schema>` (constrained decoding since v0.5).
      - Anthropic: wrapped as a tool `input_schema` with forced tool choice.
      - AirLLM   : described in the prompt only (no native schema enforcement).
    """
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": list(labels),
                "description": "Classification label.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Model confidence in this label, 0 to 1.",
            },
            "reasoning": {
                "type": "string",
                "description": "One short sentence justifying the label.",
            },
        },
        "required": ["label", "confidence", "reasoning"],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = """You are a strict text classifier. Given a short description of a memory,
decide whether it describes a CRIME under typical legal standards.

Return a JSON object with these fields (exact keys):
  label       : one of "no", "yes", "unclear"
  confidence  : float 0.0 to 1.0
  reasoning   : one short sentence

Label rules:
- "yes"     : the text clearly describes a crime (theft, fraud, assault, vandalism,
              forgery, trespass, illegal trade, etc.)
- "no"      : the text describes lawful or neutral behavior
- "unclear" : ambiguous; insufficient information; mentions a possible event but
              who-did-what-and-whether-it-was-illegal is not stated

Be strict on "yes" — require an actual unlawful act, not just suspicious tone.
Be strict on "no"  — require clearly lawful behavior.
Default to "unclear" when in doubt.
""".strip()


@dataclass
class ZeroShotResult:
    label: str
    confidence: float
    reasoning: str
    raw: str
    error: str | None = None


def _parse(raw: str, labels: Iterable[str]) -> tuple[str, float, str]:
    labels = tuple(labels)

    # Tolerate models that wrap JSON in ```json ... ``` fences.
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    # Tolerate trailing prose / repeated objects: grab first JSON object only.
    start = s.find("{")
    if start == -1:
        raise json.JSONDecodeError("no JSON object found", s, 0)
    data, _end = json.JSONDecoder().raw_decode(s[start:])
    label = str(data.get("label", "unclear")).lower().strip()
    if label not in labels:
        label = "unclear"
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    reasoning = str(data.get("reasoning", "")).strip()
    return label, conf, reasoning


class OllamaClassifier:
    """Zero-shot crime classifier backed by a local Ollama model.

    Prereqs (once):
        # Mac:  brew install ollama && ollama serve &
        # Win:  download installer from https://ollama.com/download
        ollama pull qwen2.5:3b-instruct       # ~2 GB, fast, decent
        # or:  ollama pull llama3.2:3b-instruct
        # or:  ollama pull qwen2.5:7b-instruct    # ~4.5 GB, better
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str = "http://localhost:11434",
        labels: Iterable[str] = DEFAULT_LABELS,
        temperature: float = 0.0,
        timeout: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.labels = tuple(labels)
        self.temperature = temperature
        self.timeout = timeout
        self.schema = build_output_schema(self.labels)

    def classify(self, text: str) -> ZeroShotResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            # JSON Schema constrained decoding (Ollama >= 0.5). Guarantees the
            # response matches our schema; the loose `"json"` mode is a fallback.
            "format": self.schema,
            "options": {"temperature": self.temperature},
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            label, conf, reasoning = _parse(raw, self.labels)
            return ZeroShotResult(label=label, confidence=conf, reasoning=reasoning, raw=raw)
        except Exception as e:  # noqa: BLE001
            return ZeroShotResult(
                label="unclear",
                confidence=0.0,
                reasoning="",
                raw="",
                error=f"{type(e).__name__}: {e}",
            )

    def classify_many(self, texts: list[str]) -> list[ZeroShotResult]:
        return [self.classify(t) for t in texts]


class AirLLMClassifier:
    """Zero-shot classifier backed by AirLLM (layer-by-layer disk offload).

    AirLLM loads exactly one transformer layer at a time from disk during the
    forward pass → you can run 7B-70B LLMs on tiny GPUs / Mac Silicon. Trade-off:
    slow (seconds per token, depending on layer count and disk speed).

    Platform routing:
      - Mac (Darwin): AirLLM auto-selects its MLX backend. Tokens passed as
        `mx.array(...)`. No bitsandbytes quantization (CUDA-only).
      - Windows / Linux + NVIDIA: AirLLM uses torch.cuda. 4-bit / 8-bit
        compression supported via bitsandbytes.

    Prereqs:
        uv add airllm                              # Mac
        uv add airllm bitsandbytes accelerate      # NVIDIA
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        compression: str | None = "4bit",   # "4bit" | "8bit" | None — CUDA only
        max_new_tokens: int = 200,
        max_input_tokens: int = 2048,
        labels: Iterable[str] = DEFAULT_LABELS,
        device: str | None = None,          # "cuda" | "mlx" | "cpu"; auto if None
    ):
        try:
            from airllm import AutoModel  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "airllm not installed. Run: uv add airllm "
                "(plus 'bitsandbytes accelerate' on NVIDIA for 4/8-bit)."
            ) from e
        from airllm import AutoModel

        import platform as _platform
        if device is None:
            if _platform.system() == "Darwin":
                device = "mlx"
            else:
                from .device import resolve_device
                device = resolve_device().backend  # cuda / cpu (mps not used by AirLLM)

        # bitsandbytes 4/8-bit only works on CUDA.
        if device != "cuda" and compression is not None:
            print(
                f"[crimellm] {device} backend does not support {compression!r} "
                "compression; falling back to compression=None."
            )
            compression = None

        kwargs: dict = {}
        if compression:
            kwargs["compression"] = compression
        self.model = AutoModel.from_pretrained(model_id, **kwargs)
        self.tokenizer = self.model.tokenizer
        self.labels = tuple(labels)
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.device = device
        self.model_id = model_id

        # Lazy-import the array library matching the backend.
        if self.device == "mlx":
            import mlx.core as _mx  # noqa: F401
            self._mx = _mx
        else:
            self._mx = None

    def _build_prompt(self, text: str) -> str:
        # AirLLM has no native constrained decoding. Pin the schema into the
        # prompt as an explicit JSON shape — instruction-tuned models follow it
        # and `_parse` tolerates minor wrapping (fences, whitespace).
        schema = build_output_schema(self.labels)
        user_block = (
            f"Classify this text:\n\n{text}\n\n"
            f"Respond with ONE JSON object matching this schema (no prose, no fences):\n"
            f"{json.dumps(schema)}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"{SYSTEM_PROMPT}\n\n{user_block}\n\nJSON: "

    def classify(self, text: str) -> ZeroShotResult:
        try:
            prompt = self._build_prompt(text)

            # MLX path: numpy tokens → mx.array.
            # CUDA path: torch tensor on .cuda().
            return_tensors = "np" if self.device == "mlx" else "pt"
            enc = self.tokenizer(
                prompt,
                return_tensors=return_tensors,
                truncation=True,
                max_length=self.max_input_tokens,
                return_attention_mask=False,
                padding=False,
            )
            ids = enc["input_ids"]
            input_len = ids.shape[-1]

            if self.device == "mlx":
                ids = self._mx.array(ids)
            elif self.device == "cuda":
                ids = ids.cuda()

            out = self.model.generate(
                ids,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
            )

            # AirLLM returns shapes that vary by backend / version. Normalize.
            if isinstance(out, str):
                raw = out
            else:
                seq = getattr(out, "sequences", out)
                # seq may be torch tensor, mx.array, numpy, or list
                if self.device == "mlx" and hasattr(seq, "tolist"):
                    seq_list = seq.tolist()
                    new_tokens = seq_list[0][input_len:] if isinstance(seq_list[0], list) else seq_list[input_len:]
                else:
                    new_tokens = seq[0][input_len:] if hasattr(seq, "__getitem__") else seq
                raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)


            try:
                label, conf, reasoning = _parse(raw, self.labels)
                return ZeroShotResult(label=label, confidence=conf, reasoning=reasoning, raw=raw)
            except Exception as e:  # noqa: BLE001
                return ZeroShotResult(
                    label="unclear",
                    confidence=0.0,
                    reasoning="",
                    raw=raw,
                    error=f"{type(e).__name__}: {e}",
                )
        except Exception as e:  # noqa: BLE001
            return ZeroShotResult(
                label="unclear",
                confidence=0.0,
                reasoning="",
                raw="",
                error=f"{type(e).__name__}: {e}",
            )

    def classify_many(self, texts: list[str]) -> list[ZeroShotResult]:
        return [self.classify(t) for t in texts]


class AnthropicClassifier:
    """Zero-shot crime classifier backed by Claude API.

    Prereqs:
        uv add anthropic
        export ANTHROPIC_API_KEY=sk-ant-...

    Uses prompt caching on the system prompt to cut cost when classifying
    many texts in a session (cache hit ~90% cheaper than uncached input).
    """

    TOOL_NAME = "record_crime_classification"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        labels: Iterable[str] = DEFAULT_LABELS,
        max_tokens: int = 300,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Run: uv add anthropic"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY env var not set")
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.labels = tuple(labels)
        self.max_tokens = max_tokens
        self.schema = build_output_schema(self.labels)
        self._tool = {
            "name": self.TOOL_NAME,
            "description": (
                "Record the crime-classification decision for the given text. "
                "Always call this tool exactly once."
            ),
            "input_schema": self.schema,
        }

    def classify(self, text: str) -> ZeroShotResult:
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        # Cache the static system prompt + tool schema.
                        # First call writes the cache; subsequent calls in the
                        # 5-min TTL window hit it (~90% input-token discount).
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[self._tool],
                # Force the model to use OUR tool → output is schema-validated
                # by Anthropic before we ever see it. No JSON parsing failures.
                tool_choice={"type": "tool", "name": self.TOOL_NAME},
                messages=[{"role": "user", "content": text}],
            )
            # Extract the tool_use block. With forced tool_choice it must exist.
            data = None
            for block in msg.content:
                if getattr(block, "type", None) == "tool_use" and block.name == self.TOOL_NAME:
                    data = block.input
                    break
            if data is None:
                raise RuntimeError("no tool_use block in response")

            label = str(data.get("label", "unclear")).lower().strip()
            if label not in self.labels:
                label = "unclear"
            conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            reasoning = str(data.get("reasoning", "")).strip()
            return ZeroShotResult(
                label=label,
                confidence=conf,
                reasoning=reasoning,
                raw=json.dumps(data),
            )
        except Exception as e:  # noqa: BLE001
            return ZeroShotResult(
                label="unclear",
                confidence=0.0,
                reasoning="",
                raw="",
                error=f"{type(e).__name__}: {e}",
            )

    def classify_many(self, texts: list[str]) -> list[ZeroShotResult]:
        return [self.classify(t) for t in texts]
