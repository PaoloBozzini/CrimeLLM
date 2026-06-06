from __future__ import annotations

import platform
from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    device: torch.device
    backend: str  # "cuda" | "mps" | "cpu"
    name: str
    supports_fp16: bool
    supports_bf16: bool

    def __str__(self) -> str:
        return f"{self.backend}:{self.name} (fp16={self.supports_fp16}, bf16={self.supports_bf16})"


def resolve_device() -> DeviceInfo:
    """Pick best available device. CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        idx = 0
        name = torch.cuda.get_device_name(idx)
        major, _ = torch.cuda.get_device_capability(idx)
        return DeviceInfo(
            device=torch.device(f"cuda:{idx}"),
            backend="cuda",
            name=name,
            supports_fp16=True,
            supports_bf16=major >= 8,  # Ampere+ has native bf16
        )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo(
            device=torch.device("mps"),
            backend="mps",
            name=f"Apple Silicon ({platform.machine()})",
            supports_fp16=False,  # mixed-precision on MPS is flaky for training
            supports_bf16=False,
        )
    return DeviceInfo(
        device=torch.device("cpu"),
        backend="cpu",
        name=platform.processor() or platform.machine() or "cpu",
        supports_fp16=False,
        supports_bf16=False,
    )


def training_kwargs_for_device(info: DeviceInfo | None = None) -> dict:
    """Per-backend HuggingFace TrainingArguments kwargs."""
    info = info or resolve_device()
    kwargs: dict = {}
    if info.backend == "cuda":
        # bf16 is more numerically stable, prefer when available.
        if info.supports_bf16:
            kwargs["bf16"] = True
        else:
            kwargs["fp16"] = True
        kwargs["dataloader_pin_memory"] = True
    elif info.backend == "mps":
        # Keep fp32 on MPS (mixed-precision is unstable). Trainer auto-detects MPS;
        # just disable pin_memory (CUDA-only feature).
        kwargs["dataloader_pin_memory"] = False
    else:
        kwargs["dataloader_pin_memory"] = False
    return kwargs
