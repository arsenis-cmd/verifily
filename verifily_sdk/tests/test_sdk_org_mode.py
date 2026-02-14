"""SDK tests for Org & Access v1 multi-tenant control plane.

Target: ~10 tests, runtime <0.3s
"""

from __future__ import annotations

import os
from typing import Generator

import pytest

from verifily_sdk import VerifilyClient


@pytest.fixture(autouse=True)
def _org_mode_env(monkeypatch):
    """Enable org mode for every test in this module."""
    monkeypatch.setenv("VERIFILY_ORG_MODE", "1")


@pytest.fixture
def server_url() -> str:
    """Server URL for tests."""
    return "http://127.0.0.1:8000"


@pytest.fixture
def client(server_url: str) -> VerifilyClient:
    """Create SDK client."""
    return VerifilyClient(base_url=server_url)


class TestOrgModeSDK:
    """Test SDK org mode methods (requires running server)."""
    
    def test_client_has_org_methods(self) -> None:
        """Client has org mode methods."""
        client = VerifilyClient()
        assert hasattr(client, "om_create_org")
        assert hasattr(client, "om_create_project")
        assert hasattr(client, "om_list_projects")
        assert hasattr(client, "om_create_key")
        assert hasattr(client, "om_revoke_key")
        assert hasattr(client, "om_list_keys")
    
    def test_key_format_validation(self) -> None:
        """Key manager validates key format."""
        from verifily_cli_v1.core.api.identity import KeyManager
        
        # Valid format
        valid = "vf_abcdefghijklmnopqrstuvwxyz12"
        assert KeyManager.is_valid_format(valid) is True
        
        # Invalid - no prefix
        invalid = "invalid_key"
        assert KeyManager.is_valid_format(invalid) is False
        
        # Invalid - too short
        short = "vf_abc"
        assert KeyManager.is_valid_format(short) is False
    
    def test_secret_hashing(self) -> None:
        """Secrets are hashed consistently."""
        from verifily_cli_v1.core.api.identity import KeyManager
        
        secret = "vf_test_secret_for_hashing_123"
        hash1 = KeyManager.hash_secret(secret)
        hash2 = KeyManager.hash_secret(secret)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex
    
    def test_key_id_derivation(self) -> None:
        """Key ID derived from secret is deterministic."""
        from verifily_cli_v1.core.api.identity import KeyManager
        
        secret = KeyManager.generate_secret()
        key_id = KeyManager.derive_key_id(secret)
        
        assert len(key_id) == 12
        assert key_id == KeyManager.derive_key_id(secret)
    
    def test_role_hierarchy(self) -> None:
        """Role hierarchy works correctly."""
        from verifily_cli_v1.core.api.identity import Role, has_permission
        
        # ADMIN can do everything
        assert has_permission(Role.ADMIN, Role.ADMIN) is True
        assert has_permission(Role.ADMIN, Role.DEV) is True
        assert has_permission(Role.ADMIN, Role.VIEWER) is True
        
        # DEV can do DEV and VIEWER
        assert has_permission(Role.DEV, Role.ADMIN) is False
        assert has_permission(Role.DEV, Role.DEV) is True
        assert has_permission(Role.DEV, Role.VIEWER) is True
        
        # VIEWER can only do VIEWER
        assert has_permission(Role.VIEWER, Role.ADMIN) is False
        assert has_permission(Role.VIEWER, Role.DEV) is False
        assert has_permission(Role.VIEWER, Role.VIEWER) is True


class TestOrgStoreDirect:
    """Direct tests of org store (no server required)."""
    
    def test_org_creation(self) -> None:
        """Can create org directly in store."""
        from verifily_cli_v1.core.api.org_store import OrgStore
        
        store = OrgStore()
        org = store.create_org("Test Org")
        
        assert org.name == "Test Org"
        assert org.org_id.startswith("org_")
        assert org in store.list_orgs()
    
    def test_project_creation(self) -> None:
        """Can create project in org."""
        from verifily_cli_v1.core.api.org_store import OrgStore
        
        store = OrgStore()
        org = store.create_org("Test Org")
        project = store.create_project(org.org_id, "Test Project")
        
        assert project.name == "Test Project"
        assert project.org_id == org.org_id
        assert project in store.list_projects(org_id=org.org_id)
    
    def test_key_lifecycle(self) -> None:
        """Full key lifecycle works."""
        from verifily_cli_v1.core.api.identity import Role
        from verifily_cli_v1.core.api.org_store import OrgStore
        
        store = OrgStore()
        org = store.create_org("Test Org")
        project = store.create_project(org.org_id, "Test Project")
        
        # Create key
        secret, api_key = store.create_key(
            org.org_id,
            project.project_id,
            Role.DEV,
            label="Test Key"
        )
        
        assert api_key.role == Role.DEV
        assert api_key.label == "Test Key"
        assert api_key.is_active is True
        
        # Resolve key
        resolved = store.resolve_key(secret)
        assert resolved is not None
        assert resolved.key_id == api_key.key_id
        
        # Revoke key
        revoked = store.revoke_key(api_key.key_id)
        assert revoked.is_active is False
        
        # Can no longer resolve
        resolved_after = store.resolve_key(secret)
        assert resolved_after is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
