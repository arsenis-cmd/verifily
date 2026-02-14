"""Organizations, Projects, and Memberships for Verifily.

In-memory store with optional append-only JSONL persistence.
Thread-safe.  No external dependencies.
"""

from __future__ import annotations

import enum
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("verifily.api")


# ── Enums ────────────────────────────────────────────────────────

class Role(str, enum.Enum):
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    MEMBER = "MEMBER"


# ── Data models ──────────────────────────────────────────────────

@dataclass
class Organization:
    id: str
    name: str
    created_at: float
    created_by: str  # api_key_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Project:
    id: str
    org_id: str
    name: str
    created_at: float
    created_by: str  # api_key_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Membership:
    org_id: str
    api_key_id: str
    role: Role
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d


# ── OrgStore ─────────────────────────────────────────────────────

class OrgStore:
    """Thread-safe in-memory store for orgs, projects, and memberships."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._orgs: Dict[str, Organization] = {}
        self._projects: Dict[str, Project] = {}
        self._memberships: List[Membership] = []
        self._persist_path: Optional[str] = None

    def reset(self) -> None:
        """Clear all state (used for test isolation)."""
        with self._lock:
            self._orgs.clear()
            self._projects.clear()
            self._memberships.clear()
            self._persist_path = None

    def configure_persistence(self, path: str) -> None:
        """Enable append-only JSONL persistence and replay existing events."""
        self._persist_path = path
        p = Path(path)
        if p.exists():
            self._replay(p)

    def _replay(self, path: Path) -> None:
        """Replay persisted events to restore state."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    self._apply_event(event)
                except (json.JSONDecodeError, KeyError):
                    logger.warning("orgs: skipping corrupt event line")

    def _apply_event(self, event: Dict[str, Any]) -> None:
        """Apply a single event to in-memory state."""
        kind = event["kind"]
        if kind == "org_created":
            org = Organization(
                id=event["id"], name=event["name"],
                created_at=event["created_at"], created_by=event["created_by"],
            )
            self._orgs[org.id] = org
        elif kind == "project_created":
            proj = Project(
                id=event["id"], org_id=event["org_id"], name=event["name"],
                created_at=event["created_at"], created_by=event["created_by"],
            )
            self._projects[proj.id] = proj
        elif kind == "membership_added":
            role = Role(event["role"])
            # Remove existing membership for same org+key if present
            self._memberships = [
                m for m in self._memberships
                if not (m.org_id == event["org_id"] and m.api_key_id == event["api_key_id"])
            ]
            self._memberships.append(Membership(
                org_id=event["org_id"], api_key_id=event["api_key_id"],
                role=role, created_at=event["created_at"],
            ))

    def _persist(self, event: Dict[str, Any]) -> None:
        """Append event to JSONL file if persistence is configured."""
        if not self._persist_path:
            return
        try:
            p = Path(self._persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, separators=(",", ":")) + "\n")
        except Exception as e:
            logger.warning("orgs: persist error: %s", e)

    # ── Org operations ────────────────────────────────────────────

    def create_org(self, name: str, api_key_id: str) -> Organization:
        """Create an organization. Creator is automatically added as OWNER."""
        with self._lock:
            org_id = uuid.uuid4().hex[:12]
            now = time.time()
            org = Organization(id=org_id, name=name, created_at=now, created_by=api_key_id)
            self._orgs[org_id] = org

            # Auto-add creator as OWNER
            membership = Membership(org_id=org_id, api_key_id=api_key_id, role=Role.OWNER, created_at=now)
            self._memberships.append(membership)

            self._persist({"kind": "org_created", "id": org_id, "name": name,
                           "created_at": now, "created_by": api_key_id})
            self._persist({"kind": "membership_added", "org_id": org_id,
                           "api_key_id": api_key_id, "role": "OWNER", "created_at": now})
            return org

    def list_orgs(self, api_key_id: Optional[str] = None) -> List[Organization]:
        """List organizations. If api_key_id given, only orgs the key belongs to."""
        with self._lock:
            if api_key_id is None:
                return list(self._orgs.values())
            member_org_ids = {m.org_id for m in self._memberships if m.api_key_id == api_key_id}
            return [o for o in self._orgs.values() if o.id in member_org_ids]

    def get_org(self, org_id: str) -> Optional[Organization]:
        """Get an organization by ID."""
        with self._lock:
            return self._orgs.get(org_id)

    # ── Project operations ────────────────────────────────────────

    def create_project(self, org_id: str, name: str, api_key_id: str) -> Project:
        """Create a project within an organization."""
        with self._lock:
            if org_id not in self._orgs:
                raise ValueError(f"Organization {org_id} not found.")
            proj_id = uuid.uuid4().hex[:12]
            now = time.time()
            proj = Project(id=proj_id, org_id=org_id, name=name, created_at=now, created_by=api_key_id)
            self._projects[proj_id] = proj

            self._persist({"kind": "project_created", "id": proj_id, "org_id": org_id,
                           "name": name, "created_at": now, "created_by": api_key_id})
            return proj

    def list_projects(self, org_id: Optional[str] = None) -> List[Project]:
        """List projects, optionally filtered by org_id."""
        with self._lock:
            if org_id is None:
                return list(self._projects.values())
            return [p for p in self._projects.values() if p.org_id == org_id]

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        with self._lock:
            return self._projects.get(project_id)

    # ── Membership operations ─────────────────────────────────────

    def add_membership(self, org_id: str, api_key_id: str, role: Role) -> Membership:
        """Add or update a membership. If already exists, updates role."""
        with self._lock:
            if org_id not in self._orgs:
                raise ValueError(f"Organization {org_id} not found.")
            now = time.time()
            # Remove existing membership for same org+key
            self._memberships = [
                m for m in self._memberships
                if not (m.org_id == org_id and m.api_key_id == api_key_id)
            ]
            membership = Membership(org_id=org_id, api_key_id=api_key_id, role=role, created_at=now)
            self._memberships.append(membership)

            self._persist({"kind": "membership_added", "org_id": org_id,
                           "api_key_id": api_key_id, "role": role.value, "created_at": now})
            return membership

    def list_memberships(self, org_id: str) -> List[Membership]:
        """List all memberships for an organization."""
        with self._lock:
            return [m for m in self._memberships if m.org_id == org_id]

    def check_access(self, org_id: str, api_key_id: str) -> Optional[Role]:
        """Check if api_key_id is a member of org_id. Returns Role or None."""
        with self._lock:
            for m in self._memberships:
                if m.org_id == org_id and m.api_key_id == api_key_id:
                    return m.role
            return None

    def require_project_access(self, project_id: str, api_key_id: str) -> Role:
        """Check access to a project's org. Raises PermissionError if denied."""
        with self._lock:
            proj = self._projects.get(project_id)
            if proj is None:
                raise ValueError(f"Project {project_id} not found.")
        # check_access acquires its own lock
        role = self.check_access(proj.org_id, api_key_id)
        if role is None:
            raise PermissionError(f"No access to project {project_id}.")
        return role


# ── Singleton ────────────────────────────────────────────────────

org_store = OrgStore()
