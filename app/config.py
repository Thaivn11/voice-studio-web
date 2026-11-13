"""Project and sample management for internal voice creation work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class ProjectManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.touch(exist_ok=True)

    def _load(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.db_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.db_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    def create_project(self, username: str, name: str, description: str | None = None) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Project name is required.")
        project = {
            "id": str(uuid4()),
            "name": name,
            "description": description or "",
            "owner": username,
            "samples": [],
        }
        entries = self._load()
        entries.insert(0, project)
        self._save(entries)
        return project

    def list_projects(self, username: str, role: str) -> list[dict[str, Any]]:
        entries = self._load()
        if role == "admin":
            return entries
        return [item for item in entries if item.get("owner") == username]

    def add_sample(self, project_id: str, filename: str, sample_kind: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        entries = self._load()
        for project in entries:
            if project.get("id") == project_id:
                sample = {
                    "id": str(uuid4()),
                    "filename": filename,
                    "sample_kind": sample_kind,
                    "metadata": metadata or {},
                }
                project.setdefault("samples", []).append(sample)
                self._save(entries)
                return sample
        raise ValueError("Project not found.")

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        for project in self._load():
            if project.get("id") == project_id:
                return project
        return None
