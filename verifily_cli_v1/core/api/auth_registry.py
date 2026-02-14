"""Scoped API key registry and project management for advanced auth mode.

Stores key hashes (never raw keys). Supports JSONL persistence with
append-only events and replay-on-boot. Thread-safe.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("verifily.api")

# ── Scopes ────────────────────────────────────────────────────────

SCOPES = frozenset({
    "report:read",
    "pipeline:run",
    "contamination:run",
    "jobs:submit",
    "jobs:read",
    "monitor:control",
    "usage:read",
    "admin:write",
})

# Exact-match endpoint → required scope
_ENDPOINT_SCOPES: Dict[str, str] = {
    "/v1/report": "report:read",
    "/v1/pipeline": "pipeline:run",
    "/v1/contamination": "contamination:run",
    "/v1/usage": "usage:read",
}

# Prefix-match endpoint → (method_filter, scope)
_PREFIX_SCOPES = [
    ("/v1/admin/", None, "admin:write"),
    ("/v1/billing/", "GET", "usage:read"),
    ("/v1/monitor/", None, "monitor:control"),
    ("/v1/jobs", "POST", "jobs:submit"),
    ("/v1/jobs", "GET", "jobs:read"),
]


def resolve_scope(path: str, method: str) -> Optional[str]:
    """Return required scope for an endpoint, or None for public/unscoped."""
    # Exact match first
    if path in _ENDPOINT_SCOPES:
        return _ENDPOINT_SCOPES[path]

    # Prefix match
    for prefix, method_filter, scope in _PREFIX_SCOPES:
        if path.startswith(prefix):
            if method_filter is None or method.upper() == method_filter:
                return scope

    return None


# ── Hashing ───────────────────────────────────────────────────────


def _hash_key(raw_key: str, salt: str) -> str:
    """Hash a raw API key with salt. Never store the raw key."""
    return hashlib.sha256(f"{salt}:{raw_key}".encode()).hexdigest()


def _verify_key(raw_key: str, salt: str, expected_hash: str) -> bool:
    """Constant-time comparison of key hash."""
    return hmac.compare_digest(_hash_key(raw_key, salt), expected_hash)


# ── Data Models ───────────────────────────────────────────────────


@dataclass
class ProjectRecord:
    id: str
    name: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "created_at": self.created_at}


@dataclass
class ApiKeyRecord:
    id: str
    name: str
    created_at: float
    key_hash: str
    scopes: List[str]
    projects_allowed: List[str]
    disabled: bool = False

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "key_hash": "***" if redact else self.key_hash,
            "scopes": list(self.scopes),
            "projects_allowed": list(self.projects_allowed),
            "disabled": self.disabled,
        }


# ── Registry ─────────────────────────────────────────────────────


class AuthRegistry:
    """Thread-safe registry for projects and scoped API keys."""

    def __init__(self, salt: str = "") -> None:
        self._salt = salt
        self._lock = threading.Lock()
        self._projects: Dict[str, ProjectRecord] = {}
        self._keys: Dict[str, ApiKeyRecord] = {}
        self._persist_path: Optional[str] = None

    def configure_salt(self, salt: str) -> None:
        self._salt = salt

    def configure_persistence(self, path: str) -> None:
        with self._lock:
            self._persist_path = path
        if path and Path(path).is_file():
            self._replay(path)

    def reset(self) -> None:
        with self._lock:
            self._projects.clear()
            self._keys.clear()
            self._persist_path = None
            self._salt = ""

    # ── Projects ──────────────────────────────────────────────

    def create_project(self, id: str, name: str) -> ProjectRecord:
        with self._lock:
            rec = ProjectRecord(id=id, name=name, created_at=time.time())
            self._projects[id] = rec
        self._persist({"action": "project_created", "id": id, "name": name, "created_at": rec.created_at})
        return rec

    def list_projects(self) -> List[ProjectRecord]:
        with self._lock:
            return list(self._projects.values())

    def get_project(self, project_id: str) -> Optional[ProjectRecord]:
        with self._lock:
            return self._projects.get(project_id)

    # ── Keys ──────────────────────────────────────────────────

    def create_key(
        self,
        id: str,
        name: str,
        raw_key: str,
        scopes: List[str],
        projects_allowed: List[str],
        disabled: bool = False,
    ) -> ApiKeyRecord:
        key_hash = _hash_key(raw_key, self._salt)
        with self._lock:
            rec = ApiKeyRecord(
                id=id, name=name, created_at=time.time(),
                key_hash=key_hash, scopes=list(scopes),
                projects_allowed=list(projects_allowed), disabled=disabled,
            )
            self._keys[id] = rec
        self._persist({
            "action": "key_created", "id": id, "name": name,
            "created_at": rec.created_at, "key_hash": key_hash,
            "scopes": rec.scopes, "projects_allowed": rec.projects_allowed,
            "disabled": disabled,
        })
        return rec

    def rotate_key(self, key_id: str, new_raw_key: str) -> ApiKeyRecord:
        new_hash = _hash_key(new_raw_key, self._salt)
        with self._lock:
            if key_id not in self._keys:
                raise ValueError(f"Key {key_id} not found")
            rec = self._keys[key_id]
            # Create new record since dataclass fields are not frozen
            rotated = ApiKeyRecord(
                id=rec.id, name=rec.name, created_at=rec.created_at,
                key_hash=new_hash, scopes=rec.scopes,
                projects_allowed=rec.projects_allowed, disabled=rec.disabled,
            )
            self._keys[key_id] = rotated
        self._persist({"action": "key_rotated", "id": key_id, "key_hash": new_hash})
        return rotated

    def disable_key(self, key_id: str) -> ApiKeyRecord:
        with self._lock:
            if key_id not in self._keys:
                raise ValueError(f"Key {key_id} not found")
            rec = self._keys[key_id]
            disabled = ApiKeyRecord(
                id=rec.id, name=rec.name, created_at=rec.created_at,
                key_hash=rec.key_hash, scopes=rec.scopes,
                projects_allowed=rec.projects_allowed, disabled=True,
            )
            self._keys[key_id] = disabled
        self._persist({"action": "key_disabled", "id": key_id})
        return disabled

    def list_keys(self) -> List[Dict[str, Any]]:
        """Return all keys with hashes redacted."""
        with self._lock:
            return [k.to_dict(redact=True) for k in self._keys.values()]

    def authenticate(self, raw_key: str) -> Optional[ApiKeyRecord]:
        """Authenticate a raw key. Returns record if valid, None otherwise."""
        with self._lock:
            for rec in self._keys.values():
                if rec.disabled:
                    continue
                if _verify_key(raw_key, self._salt, rec.key_hash):
                    return rec
        return None

    def has_admin_keys(self) -> bool:
        """Check if any key with admin:write scope exists."""
        with self._lock:
            return any(
                "admin:write" in k.scopes and not k.disabled
                for k in self._keys.values()
            )

    # ── Persistence ───────────────────────────────────────────

    def _persist(self, event: Dict[str, Any]) -> None:
        path = self._persist_path
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(event, separators=(",", ":")) + "\n")
                f.flush()
        except Exception:
            logger.warning("auth_registry: persist failed", exc_info=True)

    def _replay(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    evt = json.loads(line)
                    self._apply_event(evt)
        except (OSError, json.JSONDecodeError):
            logger.warning("auth_registry: replay failed", exc_info=True)

    def _apply_event(self, evt: Dict[str, Any]) -> None:
        action = evt.get("action")
        if action == "project_created":
            self._projects[evt["id"]] = ProjectRecord(
                id=evt["id"], name=evt["name"], created_at=evt["created_at"],
            )
        elif action == "key_created":
            self._keys[evt["id"]] = ApiKeyRecord(
                id=evt["id"], name=evt["name"], created_at=evt["created_at"],
                key_hash=evt["key_hash"], scopes=evt.get("scopes", []),
                projects_allowed=evt.get("projects_allowed", ["*"]),
                disabled=evt.get("disabled", False),
            )
        elif action == "key_rotated":
            if evt["id"] in self._keys:
                rec = self._keys[evt["id"]]
                self._keys[evt["id"]] = ApiKeyRecord(
                    id=rec.id, name=rec.name, created_at=rec.created_at,
                    key_hash=evt["key_hash"], scopes=rec.scopes,
                    projects_allowed=rec.projects_allowed, disabled=rec.disabled,
                )
        elif action == "key_disabled":
            if evt["id"] in self._keys:
                rec = self._keys[evt["id"]]
                self._keys[evt["id"]] = ApiKeyRecord(
                    id=rec.id, name=rec.name, created_at=rec.created_at,
                    key_hash=rec.key_hash, scopes=rec.scopes,
                    projects_allowed=rec.projects_allowed, disabled=True,
                )


# Singleton
auth_registry = AuthRegistry()
