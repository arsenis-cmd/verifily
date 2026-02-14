"""OrgStore - Multi-tenant organization and API key storage.

Thread-safe singleton with optional JSONL persistence (append-only).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from verifily_cli_v1.core.api.identity import ApiKey, KeyManager, Org, Project, Role

logger = logging.getLogger("verifily.org_store")


class OrgStoreError(Exception):
    """Base error for OrgStore operations."""
    pass


class OrgNotFoundError(OrgStoreError):
    """Organization not found."""
    pass


class ProjectNotFoundError(OrgStoreError):
    """Project not found."""
    pass


class KeyNotFoundError(OrgStoreError):
    """API key not found."""
    pass


class KeyRevokedError(OrgStoreError):
    """API key has been revoked."""
    pass


class OrgStore:
    """Thread-safe store for orgs, projects, and API keys.
    
    Supports optional JSONL persistence for audit trail.
    Events (append-only):
    - ORG_CREATED
    - PROJECT_CREATED
    - KEY_CREATED
    - KEY_REVOKED
    """
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._orgs: Dict[str, Org] = {}
        self._projects: Dict[str, Project] = {}
        self._api_keys: Dict[str, ApiKey] = {}
        self._key_hashes: Dict[str, str] = {}  # key_id -> hash (for verification)
        self._persist_path: Optional[str] = None
        self._persist_enabled: bool = False
    
    def configure(self, path: Optional[str] = None) -> None:
        """Configure persistence.
        
        If path is provided or VERIFILY_ORG_PERSIST=1, enable persistence.
        """
        persist_env = os.environ.get("VERIFILY_ORG_PERSIST", "0") == "1"
        path_env = os.environ.get("VERIFILY_ORG_LOG_PATH")
        
        self._persist_enabled = persist_env or path is not None
        
        if path:
            self._persist_path = path
        elif path_env:
            self._persist_path = path_env
        else:
            self._persist_path = str(Path("/tmp") / "verifily_org_events.jsonl")
        
        if self._persist_enabled:
            self._replay()
            logger.info("OrgStore persistence enabled: %s", self._persist_path)
    
    def _replay(self) -> None:
        """Replay events from persistence file."""
        if not self._persist_path or not Path(self._persist_path).exists():
            return
        
        try:
            with open(self._persist_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        self._apply_event(event)
                    except json.JSONDecodeError:
                        continue
            logger.info("Replayed events from %s", self._persist_path)
        except Exception as e:
            logger.warning("Failed to replay org events: %s", e)
    
    def _apply_event(self, event: dict) -> None:
        """Apply a single event to state."""
        event_type = event.get("type")
        data = event.get("data", {})
        
        if event_type == "ORG_CREATED":
            org = Org(
                org_id=data["org_id"],
                name=data["name"],
                created_at=data["created_at"],
            )
            self._orgs[org.org_id] = org
            
        elif event_type == "PROJECT_CREATED":
            project = Project(
                project_id=data["project_id"],
                org_id=data["org_id"],
                name=data["name"],
                created_at=data["created_at"],
            )
            self._projects[project.project_id] = project
            
        elif event_type == "KEY_CREATED":
            api_key = ApiKey(
                key_id=data["key_id"],
                org_id=data["org_id"],
                project_id=data["project_id"],
                role=Role(data["role"]),
                created_at=data["created_at"],
                label=data.get("label"),
            )
            self._api_keys[api_key.key_id] = api_key
            self._key_hashes[api_key.key_id] = data["key_hash"]
            
        elif event_type == "KEY_REVOKED":
            key_id = data["key_id"]
            if key_id in self._api_keys:
                # Recreate with revoked_at
                old_key = self._api_keys[key_id]
                self._api_keys[key_id] = ApiKey(
                    key_id=old_key.key_id,
                    org_id=old_key.org_id,
                    project_id=old_key.project_id,
                    role=old_key.role,
                    created_at=old_key.created_at,
                    revoked_at=data["revoked_at"],
                    label=old_key.label,
                )
    
    def _persist(self, event: dict) -> None:
        """Persist an event to file."""
        if not self._persist_enabled or not self._persist_path:
            return
        
        try:
            with open(self._persist_path, "a") as f:
                f.write(json.dumps(event, separators=(",", ":")) + "\n")
                f.flush()
        except Exception as e:
            logger.warning("Failed to persist org event: %s", e)
    
    # ── Org Operations ─────────────────────────────────────────────
    
    def create_org(self, name: str, org_id: Optional[str] = None) -> Org:
        """Create a new organization.
        
        Args:
            name: Organization name
            org_id: Optional explicit ID (generated if not provided)
            
        Returns:
            Created Org
        """
        with self._lock:
            if org_id is None:
                org_id = f"org_{uuid.uuid4().hex[:12]}"
            
            if org_id in self._orgs:
                raise OrgStoreError(f"Org {org_id} already exists")
            
            org = Org(org_id=org_id, name=name)
            self._orgs[org_id] = org
            
            self._persist({
                "type": "ORG_CREATED",
                "data": {
                    "org_id": org.org_id,
                    "name": org.name,
                    "created_at": org.created_at,
                }
            })
            
            logger.info("Created org: %s (%s)", org_id, name)
            return org
    
    def get_org(self, org_id: str) -> Org:
        """Get organization by ID."""
        with self._lock:
            if org_id not in self._orgs:
                raise OrgNotFoundError(f"Org {org_id} not found")
            return self._orgs[org_id]
    
    def list_orgs(self) -> List[Org]:
        """List all organizations."""
        with self._lock:
            return list(self._orgs.values())
    
    # ── Project Operations ─────────────────────────────────────────
    
    def create_project(
        self,
        org_id: str,
        name: str,
        project_id: Optional[str] = None,
    ) -> Project:
        """Create a new project within an organization.
        
        Args:
            org_id: Parent organization
            name: Project name
            project_id: Optional explicit ID
            
        Returns:
            Created Project
        """
        with self._lock:
            if org_id not in self._orgs:
                raise OrgNotFoundError(f"Org {org_id} not found")
            
            if project_id is None:
                project_id = f"proj_{uuid.uuid4().hex[:12]}"
            
            if project_id in self._projects:
                raise OrgStoreError(f"Project {project_id} already exists")
            
            project = Project(
                project_id=project_id,
                org_id=org_id,
                name=name,
            )
            self._projects[project_id] = project
            
            self._persist({
                "type": "PROJECT_CREATED",
                "data": {
                    "project_id": project.project_id,
                    "org_id": project.org_id,
                    "name": project.name,
                    "created_at": project.created_at,
                }
            })
            
            logger.info("Created project: %s in org %s", project_id, org_id)
            return project
    
    def get_project(self, project_id: str) -> Project:
        """Get project by ID."""
        with self._lock:
            if project_id not in self._projects:
                raise ProjectNotFoundError(f"Project {project_id} not found")
            return self._projects[project_id]
    
    def list_projects(self, org_id: Optional[str] = None) -> List[Project]:
        """List projects, optionally filtered by org."""
        with self._lock:
            projects = list(self._projects.values())
            if org_id:
                projects = [p for p in projects if p.org_id == org_id]
            return projects
    
    # ── API Key Operations ─────────────────────────────────────────
    
    def create_key(
        self,
        org_id: str,
        project_id: str,
        role: Role,
        label: Optional[str] = None,
    ) -> Tuple[str, ApiKey]:
        """Create a new API key.
        
        Args:
            org_id: Organization
            project_id: Project scope
            role: Permission level
            label: Optional label
            
        Returns:
            Tuple of (secret, ApiKey)
            
        Note:
            The secret is returned ONCE and never stored.
            Store only the hash for verification.
        """
        with self._lock:
            if org_id not in self._orgs:
                raise OrgNotFoundError(f"Org {org_id} not found")
            if project_id not in self._projects:
                raise ProjectNotFoundError(f"Project {project_id} not found")
            
            # Generate secret and derive key_id
            secret = KeyManager.generate_secret()
            key_id = KeyManager.derive_key_id(secret)
            key_hash = KeyManager.hash_secret(secret)
            
            api_key = ApiKey(
                key_id=key_id,
                org_id=org_id,
                project_id=project_id,
                role=role,
                label=label,
            )
            
            self._api_keys[key_id] = api_key
            self._key_hashes[key_id] = key_hash
            
            self._persist({
                "type": "KEY_CREATED",
                "data": {
                    "key_id": api_key.key_id,
                    "org_id": api_key.org_id,
                    "project_id": api_key.project_id,
                    "role": api_key.role.value,
                    "created_at": api_key.created_at,
                    "label": api_key.label,
                    "key_hash": key_hash,
                }
            })
            
            logger.info(
                "Created key %s for project %s with role %s",
                key_id, project_id, role.value
            )
            return secret, api_key
    
    def revoke_key(self, key_id: str) -> ApiKey:
        """Revoke an API key.
        
        Args:
            key_id: Key to revoke
            
        Returns:
            Revoked ApiKey
        """
        with self._lock:
            if key_id not in self._api_keys:
                raise KeyNotFoundError(f"Key {key_id} not found")
            
            old_key = self._api_keys[key_id]
            
            if old_key.revoked_at:
                raise KeyRevokedError(f"Key {key_id} already revoked")
            
            from datetime import datetime, timezone
            revoked_at = datetime.now(timezone.utc).isoformat()
            
            # Create new revoked key
            revoked_key = ApiKey(
                key_id=old_key.key_id,
                org_id=old_key.org_id,
                project_id=old_key.project_id,
                role=old_key.role,
                created_at=old_key.created_at,
                revoked_at=revoked_at,
                label=old_key.label,
            )
            
            self._api_keys[key_id] = revoked_key
            
            self._persist({
                "type": "KEY_REVOKED",
                "data": {
                    "key_id": key_id,
                    "revoked_at": revoked_at,
                }
            })
            
            logger.info("Revoked key: %s", key_id)
            return revoked_key
    
    def resolve_key(self, secret: str) -> Optional[ApiKey]:
        """Resolve a secret to an ApiKey.
        
        Args:
            secret: Plaintext API key secret
            
        Returns:
            ApiKey if valid and active, None otherwise
        """
        if not KeyManager.is_valid_format(secret):
            return None
        
        key_id = KeyManager.derive_key_id(secret)
        
        with self._lock:
            if key_id not in self._api_keys:
                return None
            
            stored_hash = self._key_hashes.get(key_id)
            if not stored_hash:
                return None
            
            if not KeyManager.verify_secret(secret, stored_hash):
                return None
            
            api_key = self._api_keys[key_id]
            if not api_key.is_active:
                return None
            
            return api_key
    
    def get_key(self, key_id: str) -> ApiKey:
        """Get API key by ID."""
        with self._lock:
            if key_id not in self._api_keys:
                raise KeyNotFoundError(f"Key {key_id} not found")
            return self._api_keys[key_id]
    
    def list_keys(
        self,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[ApiKey]:
        """List API keys with optional filtering."""
        with self._lock:
            keys = list(self._api_keys.values())
            if org_id:
                keys = [k for k in keys if k.org_id == org_id]
            if project_id:
                keys = [k for k in keys if k.project_id == project_id]
            return keys
    
    # ── Utility ────────────────────────────────────────────────────
    
    def reset(self) -> None:
        """Clear all state (for testing)."""
        with self._lock:
            self._orgs.clear()
            self._projects.clear()
            self._api_keys.clear()
            self._key_hashes.clear()
            self._persist_path = None
            self._persist_enabled = False


# Singleton instance
org_store = OrgStore()
