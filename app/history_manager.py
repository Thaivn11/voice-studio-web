"""JSON-backed recent generation history."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class HistoryManager:
    def __init__(self, history_path: Path, max_items: int = 12):
        self.history_path = Path(history_path)
        self.max_items = max_items
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.touch(exist_ok=True)

    def _load(self) -> list[dict[str, Any]]:
        if self.history_path.stat().st_size == 0:
            return []
        try:
            value = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.history_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_recent(self) -> list[dict[str, Any]]:
        entries = [item for item in self._load() if isinstance(item.get("output_path"), str) and Path(item["output_path"]).is_file()]
        entries = entries[: self.max_items]
        self._save(entries)
        return entries

    def record_generation(self, output_path: Path, *, mode: str, text: str, ref_audio: str | None = None, instruct: str | None = None, language: str | None = None) -> dict[str, Any]:
        entry = {
            "id": str(uuid4()),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "output_path": str(output_path),
            "mode": mode,
            "text": text[:200],
            "ref_audio": ref_audio,
            "instruct": instruct,
            "language": language,
        }
        self._save([entry, *self._load()][: self.max_items])
        return entry

    def delete_entry(self, entry_id: str) -> None:
        self._save([item for item in self._load() if item.get("id") != entry_id])
