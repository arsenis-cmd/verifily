"""WorkspacesStore: JSON-file-backed org/project/key management.

Thread-safe singleton. Nested schema:
  {"orgs": {"org_xxx": {"name": ..., "created_at": ..., "projects": {"proj_xxx": {"name": ..., "billing_plan": ..., "created_at": ..., "keys": {"<key_id>": {"key_hash": ..., "role": ..., "created_at": ..., "revoked_at": null}}}}}}}

Reuses KeyManager from core/api/identity.py for key generation and hashing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from verifily_cli_v1.core.api.identity import KeyManager, Role


# ── Role mapping ────────────────────────────────────────────────

VALID_ROLES = {"admin", "editor", "viewer"}
ROLE_MAP: Dict[str, Role] = {
    "admin": Role.ADMIN,
    "editor": Role.DEV,
    "viewer": Role.VIEWER,
}
ROLE_REVERSE: Dict[Role, str] = {v: k for k, v in ROLE_MAP.items()}


# ── Exceptions ──────────────────────────────────────────────────

class WorkspacesStoreError(Exception):
    """Base error for workspaces store operations."""


class OrgNotFoundError(WorkspacesStoreError):
    """Raised when org does not exist."""


class ProjectNotFoundError(WorkspacesStoreError):
    """Raised when project does not exist."""


class KeyNotFoundError(WorkspacesStoreError):
    """Raised when API key does not exist."""


class InvalidRoleError(WorkspacesStoreError):
    """Raised when role is not valid."""


# ── Store ───────────────────────────────────────────────────────

class WorkspacesStore:
    """Thread-safe JSON-file-backed workspace store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {"orgs": {}}
        self._path: Optional[str] = None
        self._salt: str = ""
        # key_id → (org_id, project_id, key_hash) for O(1) resolve
        self._key_index: Dict[str, Tuple[str, str, str]] = {}

    def configure(self, path: str, salt: str = "") -> None:
        """Configure persistence path and salt, then load existing data."""
        with self._lock:
            self._path = path
            self._salt = salt
            self._load()

    def reset(self) -> None:
        """Clear all data (for test isolation)."""
        with self._lock:
            self._data = {"orgs": {}}
            self._key_index = {}

    def is_empty(self) -> bool:
        """Return True if no orgs exist."""
        with self._lock:
            return len(self._data["orgs"]) == 0

    # ── CRUD ────────────────────────────────────────────────────

    def create_org(self, name: str) -> Dict[str, str]:
        """Create a new organization. Returns {org_id, name}."""
        with self._lock:
            org_id = f"org_{uuid.uuid4().hex[:12]}"
            self._data["orgs"][org_id] = {
                "name": name,
                "created_at": _now(),
                "projects": {},
            }
            self._save()
            return {"org_id": org_id, "name": name}

    def create_project(
        self, org_id: str, name: str, billing_plan: str = "free"
    ) -> Dict[str, str]:
        """Create a project under an org. Returns {project_id, org_id, name}."""
        with self._lock:
            org = self._get_org(org_id)
            project_id = f"proj_{uuid.uuid4().hex[:12]}"
            org["projects"][project_id] = {
                "name": name,
                "billing_plan": billing_plan,
                "created_at": _now(),
                "keys": {},
            }
            self._save()
            return {"project_id": project_id, "org_id": org_id, "name": name}

    def create_api_key(
        self, project_id: str, role: str
    ) -> Dict[str, str]:
        """Create an API key for a project. Returns {api_key, api_key_id, role}.

        The api_key (plaintext secret) is returned ONCE and never stored.
        """
        if role not in VALID_ROLES:
            raise InvalidRoleError(
                f"Invalid role '{role}'. Must be one of: {', '.join(sorted(VALID_ROLES))}"
            )
        with self._lock:
            org_id, project = self._get_project(project_id)
            secret = KeyManager.generate_secret()
            key_id = KeyManager.derive_key_id(secret)
            key_hash = self._hash_key(secret)
            project["keys"][key_id] = {
                "key_hash": key_hash,
                "role": role,
                "created_at": _now(),
                "revoked_at": None,
            }
            self._key_index[key_id] = (org_id, project_id, key_hash)
            self._save()
            return {"api_key": secret, "api_key_id": key_id, "role": role}

    def revoke_api_key(
        self, project_id: str, api_key_id: str
    ) -> Dict[str, Any]:
        """Revoke an API key. Returns {ok: True}."""
        with self._lock:
            _org_id, project = self._get_project(project_id)
            key_entry = project["keys"].get(api_key_id)
            if key_entry is None:
                raise KeyNotFoundError(f"Key '{api_key_id}' not found in project '{project_id}'")
            key_entry["revoked_at"] = _now()
            # Remove from key index so resolve_request fails fast
            self._key_index.pop(api_key_id, None)
            self._save()
            return {"ok": True}

    def resolve_request(self, secret: str) -> Optional[Dict[str, str]]:
        """Resolve a raw API key to {org_id, project_id, role, api_key_id}.

        Returns None if key is invalid, revoked, or unknown.
        """
        if not KeyManager.is_valid_format(secret):
            return None
        key_id = KeyManager.derive_key_id(secret)
        with self._lock:
            entry = self._key_index.get(key_id)
            if entry is None:
                return None
            org_id, project_id, stored_hash = entry
            # Constant-time comparison
            computed = self._hash_key(secret)
            if not hmac.compare_digest(computed, stored_hash):
                return None
            # Double-check not revoked (belt and suspenders)
            org = self._data["orgs"].get(org_id)
            if org is None:
                return None
            proj = org["projects"].get(project_id)
            if proj is None:
                return None
            key_data = proj["keys"].get(key_id)
            if key_data is None or key_data.get("revoked_at") is not None:
                return None
            return {
                "org_id": org_id,
                "project_id": project_id,
                "role": key_data["role"],
                "api_key_id": key_id,
            }

    # ── Internal helpers ────────────────────────────────────────

    def _get_org(self, org_id: str) -> Dict[str, Any]:
        """Get org dict or raise OrgNotFoundError."""
        org = self._data["orgs"].get(org_id)
        if org is None:
            raise OrgNotFoundError(f"Org '{org_id}' not found")
        return org

    def _get_project(self, project_id: str) -> Tuple[str, Dict[str, Any]]:
        """Find project across all orgs. Returns (org_id, project_dict)."""
        for org_id, org in self._data["orgs"].items():
            if project_id in org["projects"]:
                return org_id, org["projects"][project_id]
        raise ProjectNotFoundError(f"Project '{project_id}' not found")

    def _hash_key(self, secret: str) -> str:
        """Hash a secret with salt prefix."""
        return hashlib.sha256((self._salt + secret).encode()).hexdigest()

    def _load(self) -> None:
        """Load data from JSON file (if it exists and is non-empty)."""
        if self._path and os.path.exists(self._path) and os.path.getsize(self._path) > 0:
            with open(self._path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = {"orgs": {}}
        self._rebuild_key_index()

    def _save(self) -> None:
        """Atomically save data to JSON file."""
        if not self._path:
            return
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp_path, self._path)

    def _rebuild_key_index(self) -> None:
        """Rebuild the key_id → (org_id, project_id, key_hash) index."""
        self._key_index = {}
        for org_id, org in self._data.get("orgs", {}).items():
            for proj_id, proj in org.get("projects", {}).items():
                for key_id, key_data in proj.get("keys", {}).items():
                    if key_data.get("revoked_at") is None:
                        self._key_index[key_id] = (
                            org_id,
                            proj_id,
                            key_data["key_hash"],
                        )


def _now() -> str:
    """UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Singleton
workspaces_store = WorkspacesStore()
