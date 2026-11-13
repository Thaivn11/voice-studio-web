"""Local account store with salted password hashes and expiring bearer tokens."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import logging
import secrets
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuthManager:
    def __init__(self, db_path: Path, token_ttl_hours: int = 12):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.touch(exist_ok=True)
        self.token_ttl = timedelta(hours=token_ttl_hours)
        self._users = self._load()
        self._tokens: dict[str, tuple[str, datetime]] = {}
        self._ensure_admin()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        temporary = self.db_path.with_suffix(self.db_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._users, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.db_path)

    @staticmethod
    def _password_hash(password: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
        return f"pbkdf2_sha256${salt.hex()}${derived.hex()}"

    @staticmethod
    def _verify(password: str, encoded: str) -> bool:
        try:
            algorithm, salt_hex, digest_hex = encoded.split("$", 2)
            if algorithm != "pbkdf2_sha256":
                return False
            candidate = AuthManager._password_hash(password, bytes.fromhex(salt_hex)).split("$", 2)[2]
            return hmac.compare_digest(candidate, digest_hex)
        except (ValueError, TypeError):
            return False

    def _ensure_admin(self) -> None:
        if "admin" in self._users:
            return
        password = __import__("os").environ.get("VOICE_STUDIO_ADMIN_PASSWORD")
        if not password:
            password = secrets.token_urlsafe(18)
            logger.warning(
                "First-run administrator created. Username: admin; temporary password: %s. "
                "Set VOICE_STUDIO_ADMIN_PASSWORD before first startup to choose your own.",
                password,
            )
        self._users["admin"] = {
            "username": "admin",
            "password_hash": self._password_hash(password),
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._save()

    def create_user(self, username: str, password: str, role: str = "operator") -> dict[str, str]:
        username = username.strip()
        if not username or len(username) > 64:
            raise ValueError("Username must be between 1 and 64 characters.")
        if len(password) < 12:
            raise ValueError("Password must contain at least 12 characters.")
        if role not in {"admin", "operator"}:
            raise ValueError("Role must be admin or operator.")
        if username in self._users:
            raise ValueError("User already exists.")
        self._users[username] = {
            "username": username,
            "password_hash": self._password_hash(password),
            "role": role,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._save()
        return {"username": username, "role": role}

    def validate_password(self, username: str, password: str) -> bool:
        user = self._users.get(username)
        return bool(user and self._verify(password, user.get("password_hash", "")))

    def issue_token(self, username: str) -> str:
        if username not in self._users:
            raise ValueError("Unknown user.")
        token = secrets.token_urlsafe(32)
        self._tokens[token] = (username, datetime.now(timezone.utc) + self.token_ttl)
        return token

    def validate_token(self, token: str) -> dict[str, str] | None:
        session = self._tokens.get(token)
        if session is None:
            return None
        username, expires_at = session
        if datetime.now(timezone.utc) >= expires_at:
            self._tokens.pop(token, None)
            return None
        user = self._users.get(username)
        if user is None:
            return None
        return {"username": username, "role": user["role"]}

    def list_users(self) -> list[dict[str, str]]:
        return [
            {"username": user["username"], "role": user["role"]}
            for user in self._users.values()
        ]
