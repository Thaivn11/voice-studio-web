"""Simple JSON-backed project and voice-sample metadata store."""

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
        except (OSError, json.JSONDecodeError):
            return []
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    def _save(self, projects: list[dict[str, Any]]) -> None:
        temp = self.db_path.with_suffix(self.db_path.suffix + ".tmp")
        temp.write_text(json.dumps(projects, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.db_path)

    def create_project(self, username: str, name: str, description: str = "") -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 120:
            raise ValueError("Project name must be between 1 and 120 characters.")
        project = {
            "id": str(uuid4()),
            "name": name,
            "description": description.strip()[:500],
            "owner": username,
            "samples": [],
        }
        projects = self._load()
        projects.insert(0, project)
        self._save(projects)
        return project

    def list_projects(self, username: str, role: str) -> list[dict[str, Any]]:
        projects = self._load()
        return projects if role == "admin" else [p for p in projects if p.get("owner") == username]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        return next((p for p in self._load() if p.get("id") == project_id), None)

    def add_sample(
        self,
        project_id: str,
        filename: str,
        sample_kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        projects = self._load()
        project = next((p for p in projects if p.get("id") == project_id), None)
        if project is None:
            raise ValueError("Project not found.")
        sample = {
            "id": str(uuid4()),
            "filename": filename,
            "sample_kind": sample_kind,
            "metadata": metadata or {},
        }
        project.setdefault("samples", []).append(sample)
        self._save(projects)
        return sample
