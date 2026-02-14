"""Identity models for Org & Access v1 multi-tenant control plane.

Defines Orgs, Projects, Roles, and API Keys for SaaS-style multi-tenancy.
"""

from __future__ import annotations

import enum
import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


class Role(str, enum.Enum):
    """User roles within an organization."""
    ADMIN = "admin"
    DEV = "dev"
    VIEWER = "viewer"


@dataclass(frozen=True)
class Org:
    """An organization (tenant) in Verifily.
    
    Attributes:
        org_id: Unique organization identifier
        name: Human-readable organization name
        created_at: ISO timestamp of creation
    """
    org_id: str
    name: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class Project:
    """A project within an organization.
    
    Projects provide isolation for usage, jobs, and data.
    
    Attributes:
        project_id: Unique project identifier
        org_id: Parent organization
        name: Human-readable project name
        created_at: ISO timestamp of creation
    """
    project_id: str
    org_id: str
    name: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class ApiKey:
    """An API key for authentication and authorization.
    
    The secret itself is never stored - only its hash.
    
    Attributes:
        key_id: Deterministic ID derived from secret hash (first 12 hex chars)
        org_id: Organization this key belongs to
        project_id: Project this key is scoped to
        role: Permission level
        created_at: ISO timestamp of creation
        revoked_at: ISO timestamp of revocation (None if active)
        label: Optional human-readable label
    """
    key_id: str
    org_id: str
    project_id: str
    role: Role
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    revoked_at: Optional[str] = None
    label: Optional[str] = None
    
    @property
    def is_active(self) -> bool:
        """Check if key is active (not revoked)."""
        return self.revoked_at is None
    
    def to_dict(self, include_secret: bool = False) -> dict:
        """Convert to dictionary (never includes secret)."""
        return {
            "key_id": self.key_id,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "role": self.role.value,
            "created_at": self.created_at,
            "revoked_at": self.revoked_at,
            "label": self.label,
            "is_active": self.is_active,
        }


class KeyManager:
    """Utilities for API key lifecycle management."""
    
    PREFIX = "vf_"
    KEY_LENGTH = 32
    KEY_ID_LENGTH = 12
    
    @classmethod
    def generate_secret(cls) -> str:
        """Generate a new API key secret.
        
        Format: vf_<32_urlsafe_chars>
        Example: vf_aB3x9Kp2mNqRtUvWxYz1234567890ab
        """
        random_part = secrets.token_urlsafe(cls.KEY_LENGTH)
        # Ensure consistent length by taking first 32 chars after prefix
        return f"{cls.PREFIX}{random_part[:cls.KEY_LENGTH]}"
    
    @classmethod
    def hash_secret(cls, secret: str) -> str:
        """Hash a secret for storage (SHA-256).
        
        Never store plaintext secrets.
        """
        return hashlib.sha256(secret.encode()).hexdigest()
    
    @classmethod
    def derive_key_id(cls, secret: str) -> str:
        """Derive a deterministic key ID from secret.
        
        Uses first 12 hex chars of SHA-256 hash.
        """
        full_hash = cls.hash_secret(secret)
        return full_hash[:cls.KEY_ID_LENGTH]
    
    @classmethod
    def verify_secret(cls, secret: str, stored_hash: str) -> bool:
        """Verify a secret against stored hash (constant-time).
        
        Args:
            secret: Plaintext secret to verify
            stored_hash: Stored SHA-256 hash
            
        Returns:
            True if secret matches hash
        """
        import hmac
        computed_hash = cls.hash_secret(secret)
        return hmac.compare_digest(computed_hash, stored_hash)
    
    @classmethod
    def is_valid_format(cls, secret: str) -> bool:
        """Check if secret has valid format."""
        if not secret.startswith(cls.PREFIX):
            return False
        remaining = secret[len(cls.PREFIX):]
        return len(remaining) >= 16  # Minimum reasonable length


# Role hierarchy for permission checking
ROLE_HIERARCHY = {
    Role.VIEWER: 0,
    Role.DEV: 1,
    Role.ADMIN: 2,
}


def has_permission(user_role: Role, required_role: Role) -> bool:
    """Check if user role has at least required role permissions.
    
    Args:
        user_role: Role of the user
        required_role: Minimum required role
        
    Returns:
        True if user has sufficient permissions
    """
    return ROLE_HIERARCHY.get(user_role, -1) >= ROLE_HIERARCHY.get(required_role, 0)


def can_read(role: Role) -> bool:
    """Check if role can read data."""
    return has_permission(role, Role.VIEWER)


def can_write(role: Role) -> bool:
    """Check if role can write/submit jobs."""
    return has_permission(role, Role.DEV)


def can_admin(role: Role) -> bool:
    """Check if role can administer keys and projects."""
    return has_permission(role, Role.ADMIN)
