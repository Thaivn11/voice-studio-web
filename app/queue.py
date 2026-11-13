"""Simple in-memory queue for generation jobs and status tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class QueueJob:
    id: str
    mode: str
    text: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class GenerationQueue:
    def __init__(self) -> None:
        self._jobs: list[QueueJob] = []

    def enqueue(self, *, mode: str, text: str) -> QueueJob:
        job = QueueJob(id=f"job-{len(self._jobs) + 1}", mode=mode, text=text[:120])
        self._jobs.insert(0, job)
        return job

    def update_status(self, job_id: str, status: str) -> QueueJob | None:
        for job in self._jobs:
            if job.id == job_id:
                job.status = status
                job.updated_at = datetime.now().isoformat(timespec="seconds")
                return job
        return None

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "id": job.id,
                "mode": job.mode,
                "text": job.text,
                "status": job.status,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            }
            for job in self._jobs[:10]
        ]

    def clear_completed(self) -> None:
        self._jobs = [job for job in self._jobs if job.status not in {"completed", "failed"}]
