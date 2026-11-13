"""Application settings and writable per-user data directories."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "Voice Studio"
    model_repo: str = "k2-fsa/OmniVoice"
    output_dir_name: str = "generated"
    cache_dir_name: str = "cache"
    uploads_dir_name: str = "uploads"
    history_file_name: str = "history.json"


CONFIG = AppConfig()


def _user_data_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "VoiceStudio"
    if os.name == "darwin":
        return Path.home() / "Library" / "Application Support" / "VoiceStudio"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "voice-studio"


DATA_DIR = _user_data_dir()
OUTPUT_DIR = DATA_DIR / CONFIG.output_dir_name
CACHE_DIR = DATA_DIR / CONFIG.cache_dir_name
UPLOADS_DIR = DATA_DIR / CONFIG.uploads_dir_name
HISTORY_FILE = DATA_DIR / CONFIG.history_file_name
for folder in (OUTPUT_DIR, CACHE_DIR, UPLOADS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
HISTORY_FILE.touch(exist_ok=True)
