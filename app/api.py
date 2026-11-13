"""Local-only FastAPI web application for OmniVoice inference."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import CONFIG, HISTORY_FILE, OUTPUT_DIR, UPLOADS_DIR
from app.history_manager import HistoryManager

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
if not (STATIC_DIR / "index.html").is_file():
    raise RuntimeError(f"Missing web UI file: {STATIC_DIR / 'index.html'}")

app = FastAPI(title=CONFIG.app_name, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
history = HistoryManager(HISTORY_FILE)

_engine: Any | None = None
_engine_device: str | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_status: dict[str, Any] = {
    "state": "not_loaded",
    "device": None,
    "message": "Load the model to begin.",
}
MAX_REFERENCE_BYTES = 20 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _engine_factory():
    """Import heavyweight inference dependencies only when requested."""
    from app.core.voice_engine import VoiceEngine

    return VoiceEngine


def _safe_audio_path(filename: str) -> Path:
    root = OUTPUT_DIR.resolve()
    target = (root / filename).resolve()
    if target.parent != root or target.suffix.lower() != ".wav" or not target.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return target


def _generate(engine: Any, kwargs: dict[str, Any], filename: str) -> tuple[Path, Any]:
    # OmniVoice inference is serialized because a single model instance is shared.
    with _inference_lock:
        audio = engine.generate(**kwargs)
        output_path = engine.save_audio(audio, OUTPUT_DIR / filename)
    return output_path, audio


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    return dict(_status)


@app.post("/api/load-model")
def load_model(device: str = "auto") -> dict[str, Any]:
    global _engine, _engine_device
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise HTTPException(status_code=400, detail="Device must be auto, cpu, cuda, or mps.")

    try:
        with _model_lock:
            if _engine is None:
                _status.update(state="loading", device=None, message="Loading model; first load may download files.")
                engine = _engine_factory()(CONFIG.model_repo, device=device, load_asr=False)
                _engine = engine
                _engine_device = engine.device
            elif device not in {"auto", _engine_device}:
                raise HTTPException(
                    status_code=409,
                    detail=f"Model is already loaded on {_engine_device}; restart the server to change device.",
                )
        _status.update(state="ready", device=_engine_device, message="Model ready.")
        return dict(_status)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Model load failed")
        _status.update(state="error", device=None, message=f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=_status["message"]) from exc


@app.post("/api/generate")
async def generate(
    text: str = Form(...),
    mode: str = Form("clone"),
    language: str = Form(""),
    speed: float = Form(1.0),
    num_step: int = Form(32),
    ref_text: str = Form(""),
    instruct: str = Form(""),
    consent: bool = Form(False),
    ref_audio: UploadFile | None = File(None),
) -> dict[str, Any]:
    engine = _engine
    if engine is None:
        raise HTTPException(status_code=409, detail="Load the model before generating audio.")
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text to synthesize is required.")
    if mode not in {"clone", "design", "auto"}:
        raise HTTPException(status_code=400, detail="Unsupported generation mode.")
    if not 0.5 <= speed <= 1.5 or not 4 <= num_step <= 64:
        raise HTTPException(status_code=400, detail="Speed must be 0.5–1.5 and steps 4–64.")
    if mode == "clone" and not consent:
        raise HTTPException(status_code=400, detail="Confirm permission to use this speaker's voice.")
    if mode == "clone" and ref_audio is None:
        raise HTTPException(status_code=400, detail="Reference audio is required for voice cloning.")
    if mode == "design" and not instruct.strip():
        raise HTTPException(status_code=400, detail="Voice design instruction is required.")

    saved_reference: Path | None = None
    try:
        if ref_audio is not None:
            suffix = Path(ref_audio.filename or "").suffix.lower()
            if suffix not in ALLOWED_AUDIO_SUFFIXES:
                raise HTTPException(status_code=400, detail="Unsupported reference audio format.")
            payload = await ref_audio.read(MAX_REFERENCE_BYTES + 1)
            if not payload:
                raise HTTPException(status_code=400, detail="Reference audio is empty.")
            if len(payload) > MAX_REFERENCE_BYTES:
                raise HTTPException(status_code=413, detail="Reference audio must be at most 20 MB.")
            saved_reference = UPLOADS_DIR / f"{uuid4().hex}{suffix}"
            saved_reference.write_bytes(payload)

        kwargs: dict[str, Any] = {
            "text": text,
            "speed": speed,
            "num_step": num_step,
            "language": language.strip() or None,
        }
        if mode == "clone":
            assert saved_reference is not None
            kwargs.update(ref_audio=str(saved_reference), ref_text=ref_text.strip() or None)
        elif mode == "design":
            kwargs["instruct"] = instruct.strip()

        filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        _status.update(state="generating", device=_engine_device, message="Generating audio…")
        output_path, audio = await asyncio.to_thread(_generate, engine, kwargs, filename)
        entry = history.record_generation(
            output_path,
            mode=mode,
            text=text,
            ref_audio=None,
            instruct=instruct.strip() or None,
            language=language.strip() or None,
        )
        _status.update(state="ready", device=_engine_device, message="Audio generated.")
        return {
            "id": entry["id"],
            "filename": output_path.name,
            "audio_url": f"/api/audio/{output_path.name}",
            "duration_seconds": len(audio) / engine.sampling_rate,
            "sample_rate": engine.sampling_rate,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Speech generation failed")
        _status.update(state="ready", device=_engine_device, message=f"Generation failed: {exc}")
        raise HTTPException(status_code=500, detail=_status["message"]) from exc
    finally:
        if saved_reference is not None:
            saved_reference.unlink(missing_ok=True)
        if ref_audio is not None:
            await ref_audio.close()


@app.get("/api/history")
def get_history() -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "timestamp": item.get("timestamp"),
            "mode": item.get("mode"),
            "text": item.get("text"),
            "filename": Path(item["output_path"]).name,
            "audio_url": f"/api/audio/{Path(item['output_path']).name}",
        }
        for item in history.list_recent()
    ]


@app.delete("/api/history/{entry_id}")
def delete_history(entry_id: str) -> dict[str, bool]:
    items = history.list_recent()
    selected = next((item for item in items if item.get("id") == entry_id), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="History entry not found.")
    history.delete_entry(entry_id)
    target = Path(selected["output_path"]).resolve()
    if target.parent == OUTPUT_DIR.resolve():
        target.unlink(missing_ok=True)
    return {"deleted": True}


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    return FileResponse(_safe_audio_path(filename), media_type="audio/wav")
