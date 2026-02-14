"""TeamsStore -- thread-safe CRUD for orgs, users, memberships, projects, API keys.

Same persistence pattern as billing_store and auth_registry:
  - configure_persistence(path) -> _replay(path) on boot
  - Append-only JSONL events
  - Thread-safe with threading.Lock()
  - reset() for test isolation
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from verifily_cli_v1.core.teams.models import (
    ApiKeyRecord,
    Membership,
    Org,
    Project,
    User,
)

logger = logging.getLogger("verifily.api")

VALID_ROLES = frozenset({"owner", "admin", "member", "readonly"})


class TeamsStore:
    """Thread-safe in-memory teams store with optional JSONL persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._orgs: Dict[str, Org] = {}
        self._users: Dict[str, User] = {}
        self._memberships: List[Membership] = []
        self._projects: Dict[str, Project] = {}
        self._keys: Dict[str, ApiKeyRecord] = {}
        self._persist_path: Optional[str] = None
        self._salt: str = ""

    def configure_salt(self, salt: str) -> None:
        self._salt = salt

    def configure_persistence(self, path: Optional[str]) -> None:
        with self._lock:
            self._persist_path = path
        if path and Path(path).is_file():
            self._replay(path)

    def reset(self) -> None:
        with self._lock:
            self._orgs.clear()
            self._users.clear()
            self._memberships.clear()
            self._projects.clear()
            self._keys.clear()
            self._persist_path = None
            self._salt = ""

    # ── Hashing ───────────────────────────────────────────────

    def _hash_key(self, raw_key: str) -> str:
        return hashlib.sha256(f"{self._salt}:{raw_key}".encode()).hexdigest()

    def _verify_key(self, raw_key: str, expected_hash: str) -> bool:
        return hmac.compare_digest(self._hash_key(raw_key), expected_hash)

    # ── CRUD ──────────────────────────────────────────────────

    def create_org(self, *, id: str, name: str) -> Org:
        org = Org(id=id, name=name, created_at=time.time())
        with self._lock:
            self._orgs[id] = org
        self._persist({"action": "org_created", **org.to_dict()})
        return org

    def get_org(self, org_id: str) -> Optional[Org]:
        with self._lock:
            return self._orgs.get(org_id)

    def create_user(self, *, id: str, email: str, name: str) -> User:
        user = User(id=id, email=email, name=name, created_at=time.time())
        with self._lock:
            self._users[id] = user
        self._persist({"action": "user_created", **user.to_dict()})
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        with self._lock:
            return self._users.get(user_id)

    def add_membership(self, *, user_id: str, org_id: str, role: str = "member") -> Membership:
        role = role.lower()
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {sorted(VALID_ROLES)}")
        mem = Membership(user_id=user_id, org_id=org_id, role=role, created_at=time.time())
        with self._lock:
            self._memberships.append(mem)
        self._persist({"action": "membership_added", **mem.to_dict()})
        return mem

    def get_memberships(self, *, user_id: Optional[str] = None, org_id: Optional[str] = None) -> List[Membership]:
        with self._lock:
            result = []
            for m in self._memberships:
                if user_id and m.user_id != user_id:
                    continue
                if org_id and m.org_id != org_id:
                    continue
                result.append(m)
            return result

    def create_project(self, *, id: str, org_id: str, name: str) -> Project:
        proj = Project(id=id, org_id=org_id, name=name, created_at=time.time())
        with self._lock:
            self._projects[id] = proj
        self._persist({"action": "project_created", **proj.to_dict()})
        return proj

    def get_project(self, project_id: str) -> Optional[Project]:
        with self._lock:
            return self._projects.get(project_id)

    def list_projects(self) -> List[Project]:
        with self._lock:
            return list(self._projects.values())

    def create_api_key(
        self,
        *,
        id: str,
        org_id: str,
        name: str,
        raw_key: str,
        scopes: List[str],
        project_ids: List[str],
        created_by: str,
    ) -> ApiKeyRecord:
        key_hash = self._hash_key(raw_key)
        rec = ApiKeyRecord(
            id=id,
            org_id=org_id,
            name=name,
            key_hash=key_hash,
            scopes=list(scopes),
            project_ids=list(project_ids),
            created_at=time.time(),
            created_by=created_by,
        )
        with self._lock:
            self._keys[id] = rec
        self._persist({
            "action": "key_created",
            "id": id,
            "org_id": org_id,
            "name": name,
            "key_hash": key_hash,
            "scopes": rec.scopes,
            "project_ids": rec.project_ids,
            "created_at": rec.created_at,
            "created_by": created_by,
        })
        return rec

    def resolve_key(self, raw_key: str) -> Optional[ApiKeyRecord]:
        """Authenticate a raw API key. Returns record or None."""
        with self._lock:
            for rec in self._keys.values():
                if rec.disabled:
                    continue
                if self._verify_key(raw_key, rec.key_hash):
                    return rec
        return None

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
            logger.warning("teams_store: persist failed", exc_info=True)

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
            logger.warning("teams_store: replay failed", exc_info=True)

    def _apply_event(self, evt: Dict[str, Any]) -> None:
        action = evt.get("action")
        if action == "org_created":
            self._orgs[evt["id"]] = Org(
                id=evt["id"], name=evt["name"], created_at=evt["created_at"],
            )
        elif action == "user_created":
            self._users[evt["id"]] = User(
                id=evt["id"], email=evt["email"], name=evt["name"],
                created_at=evt["created_at"], disabled=evt.get("disabled", False),
            )
        elif action == "membership_added":
            self._memberships.append(Membership(
                user_id=evt["user_id"], org_id=evt["org_id"],
                role=evt["role"], created_at=evt["created_at"],
            ))
        elif action == "project_created":
            self._projects[evt["id"]] = Project(
                id=evt["id"], org_id=evt["org_id"], name=evt["name"],
                created_at=evt["created_at"],
            )
        elif action == "key_created":
            self._keys[evt["id"]] = ApiKeyRecord(
                id=evt["id"], org_id=evt["org_id"], name=evt["name"],
                key_hash=evt["key_hash"], scopes=evt.get("scopes", []),
                project_ids=evt.get("project_ids", []),
                created_at=evt["created_at"],
                created_by=evt.get("created_by", ""),
            )


teams_store = TeamsStore()
