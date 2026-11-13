"""Local-only FastAPI app for authorized OmniVoice inference."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.auth import AuthManager
from app.config import AUTH_FILE, CONFIG, HISTORY_FILE, OUTPUT_DIR, PROJECTS_FILE, UPLOADS_DIR
from app.history_manager import HistoryManager
from app.project_manager import ProjectManager
from app.queue import GenerationQueue

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(title=CONFIG.app_name, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
history = HistoryManager(HISTORY_FILE)
queue = GenerationQueue()
auth_manager = AuthManager(AUTH_FILE)
project_manager = ProjectManager(PROJECTS_FILE)

_engine: Any | None = None
_engine_device: str | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_status: dict[str, Any] = {"state": "not_loaded", "device": None, "message": "Load the model to begin."}
MAX_REFERENCE_BYTES = 20 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _engine_factory():
    from app.core.voice_engine import VoiceEngine

    return VoiceEngine


def _current_user(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in is required.")
    user = auth_manager.validate_token(authorization[7:].strip())
    if user is None:
        raise HTTPException(status_code=401, detail="Session expired; sign in again.")
    return user


def _admin_only(user: dict[str, str]) -> None:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required.")


def _safe_audio_path(filename: str) -> Path:
    root = OUTPUT_DIR.resolve()
    target = (root / filename).resolve()
    if target.parent != root or target.suffix.lower() != ".wav" or not target.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return target


def _generate(engine: Any, kwargs: dict[str, Any], filename: str) -> tuple[Path, Any]:
    with _inference_lock:
        audio = engine.generate(**kwargs)
        path = engine.save_audio(audio, OUTPUT_DIR / filename)
    return path, audio


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login")
def login(username: str = Form(...), password: str = Form(...)) -> dict[str, str]:
    if not auth_manager.validate_password(username.strip(), password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = auth_manager.issue_token(username.strip())
    user = next(u for u in auth_manager.list_users() if u["username"] == username.strip())
    return {"token": token, **user}


@app.get("/api/auth/me")
def who_am_i(user: dict[str, str] = Depends(_current_user)) -> dict[str, str]:
    return user


@app.get("/api/status")
def get_status(user: dict[str, str] = Depends(_current_user)) -> dict[str, Any]:
    return dict(_status)


@app.get("/api/overview")
def overview(user: dict[str, str] = Depends(_current_user)) -> dict[str, Any]:
    return {
        "model_loaded": _engine is not None,
        "state": _status["state"],
        "device": _engine_device,
        "history_count": len(history.list_recent()),
        "queue_length": len(queue.list()),
        "message": _status["message"],
    }


@app.get("/api/jobs")
def get_jobs(user: dict[str, str] = Depends(_current_user)) -> list[dict[str, Any]]:
    return queue.list()


@app.get("/api/users")
def get_users(user: dict[str, str] = Depends(_current_user)) -> list[dict[str, str]]:
    _admin_only(user)
    return auth_manager.list_users()


@app.post("/api/users")
def create_user(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("operator"),
    user: dict[str, str] = Depends(_current_user),
) -> dict[str, str]:
    _admin_only(user)
    try:
        return auth_manager.create_user(username, password, role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects")
def get_projects(user: dict[str, str] = Depends(_current_user)) -> list[dict[str, Any]]:
    return project_manager.list_projects(user["username"], user["role"])


@app.post("/api/projects")
def create_project(
    name: str = Form(...),
    description: str = Form(""),
    user: dict[str, str] = Depends(_current_user),
) -> dict[str, Any]:
    try:
        return project_manager.create_project(user["username"], name, description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/samples")
async def add_sample(
    project_id: str,
    sample_kind: str = Form("voice"),
    metadata: str = Form("{}"),
    audio: UploadFile | None = File(None),
    user: dict[str, str] = Depends(_current_user),
) -> dict[str, Any]:
    project = project_manager.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    if user["role"] != "admin" and project["owner"] != user["username"]:
        raise HTTPException(status_code=403, detail="You cannot edit this project.")
    if audio is None:
        raise HTTPException(status_code=400, detail="Audio sample is required.")
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported audio file format.")
    payload = await audio.read(MAX_REFERENCE_BYTES + 1)
    if not payload or len(payload) > MAX_REFERENCE_BYTES:
        raise HTTPException(status_code=413, detail="Sample must be non-empty and at most 20 MB.")
    try:
        parsed_metadata = json.loads(metadata) if metadata else {}
        if not isinstance(parsed_metadata, dict):
            raise ValueError("Metadata must be a JSON object.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Metadata must be a JSON object.") from exc
    filename = f"{uuid4().hex}{suffix}"
    (UPLOADS_DIR / filename).write_bytes(payload)
    try:
        return project_manager.add_sample(project_id, filename, sample_kind, parsed_metadata)
    finally:
        await audio.close()


@app.post("/api/load-model")
def load_model(device: str = "auto", user: dict[str, str] = Depends(_current_user)) -> dict[str, Any]:
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
                raise HTTPException(status_code=409, detail=f"Model is already loaded on {_engine_device}; restart to change device.")
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
    user: dict[str, str] = Depends(_current_user),
) -> dict[str, Any]:
    engine = _engine
    if engine is None:
        raise HTTPException(status_code=409, detail="Load the model before generating audio.")
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    if mode not in {"clone", "design", "auto"}:
        raise HTTPException(status_code=400, detail="Unsupported mode.")
    if not 0.5 <= speed <= 1.5 or not 4 <= num_step <= 64:
        raise HTTPException(status_code=400, detail="Speed must be 0.5–1.5 and steps 4–64.")
    if mode == "clone" and not consent:
        raise HTTPException(status_code=400, detail="Confirm permission to use this speaker's voice.")
    if mode == "clone" and ref_audio is None:
        raise HTTPException(status_code=400, detail="Reference audio is required for cloning.")
    if mode == "design" and not instruct.strip():
        raise HTTPException(status_code=400, detail="Voice design instruction is required.")

    job = queue.enqueue(mode=mode, text=text)
    saved_reference: Path | None = None
    try:
        if ref_audio is not None:
            suffix = Path(ref_audio.filename or "").suffix.lower()
            if suffix not in ALLOWED_AUDIO_SUFFIXES:
                raise HTTPException(status_code=400, detail="Unsupported audio format.")
            payload = await ref_audio.read(MAX_REFERENCE_BYTES + 1)
            if not payload:
                raise HTTPException(status_code=400, detail="Reference audio is empty.")
            if len(payload) > MAX_REFERENCE_BYTES:
                raise HTTPException(status_code=413, detail="Reference audio exceeds 20 MB.")
            saved_reference = UPLOADS_DIR / f"{uuid4().hex}{suffix}"
            saved_reference.write_bytes(payload)

        queue.update_status(job.id, "running")
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
            output_path, mode=mode, text=text, ref_audio=None,
            instruct=instruct.strip() or None, language=language.strip() or None,
        )
        queue.update_status(job.id, "completed")
        _status.update(state="ready", device=_engine_device, message="Audio generated.")
        return {
            "id": entry["id"],
            "filename": output_path.name,
            "audio_url": f"/api/audio/{output_path.name}",
            "duration_seconds": len(audio) / engine.sampling_rate,
            "sample_rate": engine.sampling_rate,
        }
    except HTTPException:
        queue.update_status(job.id, "failed")
        raise
    except Exception as exc:
        queue.update_status(job.id, "failed")
        logger.exception("Speech generation failed")
        _status.update(state="ready", device=_engine_device, message=f"Generation failed: {exc}")
        raise HTTPException(status_code=500, detail=_status["message"]) from exc
    finally:
        if saved_reference is not None:
            saved_reference.unlink(missing_ok=True)
        if ref_audio is not None:
            await ref_audio.close()


@app.get("/api/history")
def get_history(user: dict[str, str] = Depends(_current_user)) -> list[dict[str, Any]]:
    return [
        {"id": item["id"], "timestamp": item.get("timestamp"), "mode": item.get("mode"),
         "text": item.get("text"), "filename": Path(item["output_path"]).name,
         "audio_url": f"/api/audio/{Path(item['output_path']).name}"}
        for item in history.list_recent()
    ]


@app.delete("/api/history/{entry_id}")
def delete_history(entry_id: str, user: dict[str, str] = Depends(_current_user)) -> dict[str, bool]:
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
def get_audio(filename: str, user: dict[str, str] = Depends(_current_user)) -> FileResponse:
    return FileResponse(_safe_audio_path(filename), media_type="audio/wav")
