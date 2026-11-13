"""Thin wrapper around the OmniVoice inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from app.core.device_manager import resolve_device


class VoiceEngine:
    def __init__(self, model_repo: str, device: str = "auto", *, load_asr: bool = False):
        try:
            from omnivoice import OmniVoice
        except ImportError as exc:
            raise RuntimeError(
                "OmniVoice is not installed. Install it with: "
                "python -m pip install git+https://github.com/k2-fsa/OmniVoice.git"
            ) from exc
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = OmniVoice.from_pretrained(
            model_repo, device_map=self.device, dtype=dtype, load_asr=load_asr
        )
        self.sampling_rate = int(self.model.sampling_rate)

    def generate(
        self,
        text: str,
        *,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
        language: str | None = None,
        speed: float = 1.0,
        num_step: int = 32,
    ) -> np.ndarray:
        if not text.strip():
            raise ValueError("Please enter text to synthesize.")
        kwargs: dict[str, Any] = {"text": text.strip(), "speed": float(speed), "num_step": int(num_step)}
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
        if instruct:
            kwargs["instruct"] = instruct.strip()
        if language:
            kwargs["language"] = language.strip()
        result = self.model.generate(**kwargs)
        if not result:
            raise RuntimeError("OmniVoice returned no audio.")
        return np.asarray(result[0], dtype=np.float32).squeeze()

    def save_audio(self, audio: np.ndarray, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, audio, self.sampling_rate, subtype="PCM_16")
        return destination
