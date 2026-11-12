"""Detect the best available device for speech generation."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _has_mps() -> bool:
    try:
        import torch

        return hasattr(torch, "mps") and bool(torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_device(preferred: str | None = None) -> str:
    if preferred:
        preferred = preferred.strip().lower()
        if preferred in {"cuda", "gpu", "auto"} and _has_cuda():
            return "cuda"
        if preferred in {"mps", "metal"} and _has_mps():
            return "mps"
        if preferred in {"cpu", "cpu-only"}:
            return "cpu"

    if _has_cuda():
        return "cuda"
    if _has_mps():
        return "mps"
    return "cpu"


def get_best_device() -> str:
    device = resolve_device()
    logger.info("Selected device: %s", device)
    return device
