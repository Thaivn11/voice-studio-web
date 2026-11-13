"""Device discovery and validation for CPU, CUDA, and Apple Metal."""

from __future__ import annotations


def available_devices() -> list[str]:
    try:
        import torch
    except ImportError:
        return ["cpu"]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def get_best_device() -> str:
    return available_devices()[0]


def resolve_device(preferred: str | None = None) -> str:
    choice = (preferred or "auto").strip().lower()
    choice = {"gpu": "cuda", "metal": "mps", "cpu-only": "cpu"}.get(choice, choice)
    devices = available_devices()
    if choice == "auto":
        return devices[0]
    if choice not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {choice}")
    if choice not in devices:
        raise RuntimeError(f"{choice} is unavailable. Available devices: {', '.join(devices)}")
    return choice
