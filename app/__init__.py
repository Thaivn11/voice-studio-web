"""FastAPI entry point for the internal web app."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import CONFIG, HISTORY_FILE, OUTPUT_DIR, UPLOADS_DIR
from app.core.device_manager import get_best_device
from app.core.voice_engine import VoiceEngine
from app.history_manager import HistoryManager

app = FastAPI(title=CONFIG.app_name, docs_url=None, redoc_url=None)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
history = HistoryManager(HISTORY_FILE)

_engine: VoiceEngine | None = None
_engine_device: str | None = None
_process_lock = asyncio.Lock()
_status: dict[str, Any] = {
    "state": "not_loaded",
    "device": None,
    "message": "Load the model to begin.",
}

ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_REFERENCE_BYTES = 20 * 1024 * 1024


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html", media_type="text/html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    return dict(_status)


@app.post("/api/load-model")
def load_model(device: str = "auto") -> dict[str, Any]:
    global _engine, _engine_device
    selected = device.strip().lower() if device else "auto"
    if selected not in {"auto", "cpu", "cuda", "mps"}:
        raise HTTPException(status_code=400, detail="Unsupported device selection.")

    if _engine is not None:
        _status.update(state="ready", device=_engine_device, message="Model already loaded.")
        return dict(_status)

    _status.update(state="loading", device=None, message="Loading OmniVoice model…")
    try:
        resolved_device = get_best_device() if selected == "auto" else selected
        _engine = VoiceEngine(CONFIG.model_repo, device=resolved_device, load_asr=False)
        _engine_device = _engine.device
        _status.update(state="ready", device=_engine_device, message="Model ready.")
        return dict(_status)
    except Exception as exc:
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
    if _engine is None:
        raise HTTPException(status_code=409, detail="Load the model before generating audio.")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text to synthesize is required.")

    if mode not in {"clone", "design", "auto"}:
        raise HTTPException(status_code=400, detail="Unsupported mode.")

    if mode == "clone":
        if not consent:
            raise HTTPException(status_code=400, detail="Consent is required for voice cloning.")
        if ref_audio is None:
            raise HTTPException(status_code=400, detail="Reference audio is required for cloning.")

    if mode == "design" and not instruct.strip():
        raise HTTPException(status_code=400, detail="Voice design instruction is required.")

    saved_reference: Path | None = None
    try:
        if ref_audio is not None:
            suffix = Path(ref_audio.filename or "").suffix.lower()
            if suffix not in ALLOWED_AUDIO_SUFFIXES:
                raise HTTPException(status_code=400, detail="Unsupported reference audio type.")
            payload = await ref_audio.read(MAX_REFERENCE_BYTES + 1)
            if len(payload) > MAX_REFERENCE_BYTES:
                raise HTTPException(status_code=413, detail="Reference audio must be under 20 MB.")
            saved_reference = UPLOADS_DIR / f"{uuid4().hex}{suffix}"
            saved_reference.write_bytes(payload)

        kwargs: dict[str, Any] = {
            "text": text,
            "speed": float(speed),
            "num_step": int(num_step),
            "language": language.strip() or None,
        }
        if mode == "clone":
            kwargs["ref_audio"] = str(saved_reference)
            kwargs["ref_text"] = ref_text.strip() or None
        elif mode == "design":
            kwargs["instruct"] = instruct.strip()

        _status.update(state="generating", device=_engine_device, message="Generating audio…")
        async with _process_lock:
            audio = await asyncio.to_thread(_engine.generate, **kwargs)
            filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            output_path = await asyncio.to_thread(_engine.save_audio, audio, OUTPUT_DIR / filename)

        history_entry = history.record_generation(
            output_path,
            mode=mode,
            text=text,
            ref_audio=str(saved_reference) if saved_reference else None,
            instruct=instruct.strip() or None,
            language=language.strip() or None,
        )

        _status.update(state="ready", device=_engine_device, message="Audio generated.")
        return {
            "id": history_entry["id"],
            "filename": output_path.name,
            "audio_url": f"/api/audio/{output_path.name}",
            "duration_seconds": len(audio) / _engine.sampling_rate,
            "sample_rate": _engine.sampling_rate,
        }
    except HTTPException:
        raise
    except Exception as exc:
        _status.update(state="error", device=_engine_device, message=f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=_status["message"]) from exc
    finally:
        if saved_reference is not None:
            saved_reference.unlink(missing_ok=True)
        if ref_audio is not None:
            await ref_audio.close()


@app.get("/api/history")
def get_history() -> list[dict[str, Any]]:
    items = history.list_recent()
    return [
        {
            "id": item["id"],
            "timestamp": item.get("timestamp"),
            "mode": item.get("mode"),
            "text": item.get("text"),
            "filename": Path(item["output_path"]).name,
            "audio_url": f"/api/audio/{Path(item['output_path']).name}",
        }
        for item in items
    ]


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    target = OUTPUT_DIR / filename
    if not target.is_file() or target.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(target, media_type="audio/wav", filename=target.name)


@app.delete("/api/history/{entry_id}")
def delete_history(entry_id: str) -> dict[str, bool]:
    items = history.list_recent()
    selected = next((item for item in items if item.get("id") == entry_id), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="History item not found.")
    history.delete_entry(entry_id)
    target = Path(selected["output_path"]).resolve()
    if target.parent == OUTPUT_DIR.resolve():
        target.unlink(missing_ok=True)
    return {"deleted": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, log_level="info")
