"""Authentication and access controls for local internal use."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class User:
    username: str
    password_hash: str
    role: str = "operator"
    token: str | None = None
    created_at: str = field(default_factory=lambda: __import__("datetime").datetime.now().isoformat(timespec="seconds"))


class AuthManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.touch(exist_ok=True)
        self._users = self._load()
        self._ensure_default_admin()

    def _hash(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return raw

    def _save(self) -> None:
        self.db_path.write_text(json.dumps(self._users, indent=2, ensure_ascii=False), encoding="utf-8")

    def _ensure_default_admin(self) -> None:
        if "admin" not in self._users:
            user = User(username="admin", password_hash=self._hash("admin123"), role="admin")
            self._users["admin"] = user.__dict__
            self._save()

    def create_user(self, username: str, password: str, role: str = "operator") -> User:
        username = username.strip()
        if not username or not password:
            raise ValueError("Username and password are required.")
        if username in self._users:
            raise ValueError("User already exists.")
        user = User(username=username, password_hash=self._hash(password), role=role)
        self._users[username] = user.__dict__
        self._save()
        return user

    def validate_password(self, username: str, password: str) -> bool:
        user = self._users.get(username)
        if not user:
            return False
        return user.get("password_hash") == self._hash(password)

    def issue_token(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        user = self._users.get(username)
        if not user:
            raise ValueError("Unknown user.")
        user["token"] = token
        self._save()
        return token

    def validate_token(self, token: str) -> dict[str, Any] | None:
        for user in self._users.values():
            if user.get("token") == token:
                return {"username": user["username"], "role": user["role"]}
        return None

    def list_users(self) -> list[dict[str, Any]]:
        return [{"username": user["username"], "role": user["role"]} for user in self._users.values()]
