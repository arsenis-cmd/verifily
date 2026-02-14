"""Teams data models -- Org, User, Membership, Project, ApiKeyRecord.

All models are plain dataclasses with to_dict() for serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Org:
    """An organization (tenant)."""

    id: str
    name: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "created_at": self.created_at}


@dataclass
class User:
    """A named user within the teams system."""

    id: str
    email: str
    name: str
    created_at: float
    disabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at,
            "disabled": self.disabled,
        }


@dataclass
class Membership:
    """Links a user to an organization with a role."""

    user_id: str
    org_id: str
    role: str  # "owner" | "admin" | "member" | "readonly"
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "role": self.role,
            "created_at": self.created_at,
        }


@dataclass
class Project:
    """An org-scoped project."""

    id: str
    org_id: str
    name: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "name": self.name,
            "created_at": self.created_at,
        }


@dataclass
class ApiKeyRecord:
    """A scoped API key bound to an org and optional projects."""

    id: str
    org_id: str
    name: str
    key_hash: str
    scopes: List[str] = field(default_factory=list)
    project_ids: List[str] = field(default_factory=list)
    created_at: float = 0.0
    created_by: str = ""
    disabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "name": self.name,
            "scopes": list(self.scopes),
            "project_ids": list(self.project_ids),
            "created_at": self.created_at,
            "created_by": self.created_by,
            "disabled": self.disabled,
        }
