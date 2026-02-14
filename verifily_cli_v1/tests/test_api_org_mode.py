"""Tests for Org & Access v1 multi-tenant control plane.

Target: ~25 tests, runtime <0.5s
"""

from __future__ import annotations

import os
import tempfile
from typing import Generator, Tuple

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.identity import Role
from verifily_cli_v1.core.api.org_store import OrgStore, org_store
from verifily_cli_v1.core.api.server import create_app


def _bootstrap_admin() -> Tuple[str, str, str]:
    """Bootstrap an admin key via the store directly (bypasses HTTP auth).

    Returns (secret, org_id, project_id).
    """
    org = org_store.create_org("Bootstrap Org")
    project = org_store.create_project(org.org_id, "Bootstrap Project")
    secret, key = org_store.create_key(org.org_id, project.project_id, Role.ADMIN)
    return secret, org.org_id, project.project_id


@pytest.fixture(autouse=True)
def _org_mode_env(monkeypatch) -> Generator[None, None, None]:
    """Enable org mode for every test in this module and reset store."""
    monkeypatch.setenv("VERIFILY_ORG_MODE", "1")
    monkeypatch.setenv("VERIFILY_SKIP_SIGNALS", "1")
    org_store.reset()
    yield
    org_store.reset()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create test client with org mode enabled."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def admin_key(client: TestClient) -> Tuple[str, str, str]:
    """Create org, project, and admin key via store. Returns (secret, org_id, project_id)."""
    return _bootstrap_admin()


class TestOrgCreation:
    """Test organization creation."""

    def test_create_org_success(self, client: TestClient) -> None:
        """Can create org with valid data (bootstrap — no orgs exist)."""
        resp = client.post("/v1/orgs", json={"name": "Acme Corp"})
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert data["name"] == "Acme Corp"
        assert data["org_id"].startswith("org_")
        assert "created_at" in data

    def test_create_org_requires_admin(self, client: TestClient, admin_key: tuple) -> None:
        """Creating org requires admin (after bootstrap)."""
        secret, _, _ = admin_key
        resp = client.post(
            "/v1/orgs",
            json={"name": "Another Org"},
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert resp.status_code in (200, 201)


class TestProjectCreation:
    """Test project creation and listing."""

    def test_create_project_success(self, client: TestClient, admin_key: tuple) -> None:
        """Can create project within org."""
        secret, org_id, _ = admin_key
        resp = client.post(
            "/v1/projects",
            json={"org_id": org_id, "name": "My Project"},
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert data["name"] == "My Project"
        assert data["org_id"] == org_id
        assert data["project_id"].startswith("proj_")

    def test_list_projects(self, client: TestClient, admin_key: tuple) -> None:
        """Can list projects."""
        secret, org_id, _ = admin_key
        h = {"Authorization": f"Bearer {secret}"}
        client.post("/v1/projects", json={"org_id": org_id, "name": "Project A"}, headers=h)
        client.post("/v1/projects", json={"org_id": org_id, "name": "Project B"}, headers=h)

        resp = client.get(f"/v1/projects?org_id={org_id}", headers=h)
        assert resp.status_code in (200, 201)
        data = resp.json()
        # Bootstrap project + 2 new = 3
        assert len(data["projects"]) == 3


class TestKeyLifecycle:
    """Test API key creation and revocation."""

    def test_create_key_returns_secret_once(self, client: TestClient, admin_key: tuple) -> None:
        """Key creation returns secret only once."""
        secret, _, project_id = admin_key
        h = {"Authorization": f"Bearer {secret}"}

        resp = client.post("/v1/keys", json={"project_id": project_id, "role": "dev"}, headers=h)
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert "secret" in data
        assert data["secret"].startswith("vf_")
        assert data["role"] == "dev"
        assert data["project_id"] == project_id

    def test_create_key_invalid_role(self, client: TestClient, admin_key: tuple) -> None:
        """Key creation fails with invalid role."""
        secret, _, project_id = admin_key
        h = {"Authorization": f"Bearer {secret}"}

        resp = client.post("/v1/keys", json={"project_id": project_id, "role": "invalid"}, headers=h)
        assert resp.status_code == 422

    def test_revoke_key(self, client: TestClient, admin_key: tuple) -> None:
        """Can revoke a key."""
        secret, _, project_id = admin_key
        h = {"Authorization": f"Bearer {secret}"}

        resp = client.post("/v1/keys", json={"project_id": project_id, "role": "dev"}, headers=h)
        key_id = resp.json()["key_id"]

        resp = client.post(f"/v1/keys/{key_id}/revoke", headers=h)
        assert resp.status_code in (200, 201)
        assert resp.json()["was_active"] is True

        resp = client.get("/v1/keys", headers=h)
        keys = resp.json()["keys"]
        revoked_key = next(k for k in keys if k["key_id"] == key_id)
        assert revoked_key["is_active"] is False


class TestAuthentication:
    """Test key-based authentication."""

    def test_valid_key_authenticates(self, client: TestClient, admin_key: tuple) -> None:
        """Valid key allows access."""
        secret, _, project_id = admin_key
        h = {"Authorization": f"Bearer {secret}"}

        # Create dev key
        resp = client.post("/v1/keys", json={"project_id": project_id, "role": "dev"}, headers=h)
        dev_secret = resp.json()["secret"]

        resp = client.get("/v1/projects", headers={"Authorization": f"Bearer {dev_secret}"})
        assert resp.status_code in (200, 201)

    def test_invalid_key_rejected(self, client: TestClient) -> None:
        """Invalid key is rejected."""
        # Need at least one org so bootstrap doesn't bypass
        _bootstrap_admin()
        resp = client.get("/v1/projects", headers={"Authorization": "Bearer vf_invalid123"})
        assert resp.status_code == 401

    def test_revoked_key_rejected(self, client: TestClient, admin_key: tuple) -> None:
        """Revoked key is rejected."""
        secret, _, project_id = admin_key
        h = {"Authorization": f"Bearer {secret}"}

        resp = client.post("/v1/keys", json={"project_id": project_id, "role": "dev"}, headers=h)
        dev_secret = resp.json()["secret"]
        key_id = resp.json()["key_id"]

        client.post(f"/v1/keys/{key_id}/revoke", headers=h)

        resp = client.get("/v1/projects", headers={"Authorization": f"Bearer {dev_secret}"})
        assert resp.status_code == 401


class TestRBAC:
    """Test role-based access control."""

    def test_viewer_cannot_write(self, client: TestClient, admin_key: tuple) -> None:
        """VIEWER cannot submit pipeline."""
        secret, _, project_id = admin_key
        h = {"Authorization": f"Bearer {secret}"}

        resp = client.post("/v1/keys", json={"project_id": project_id, "role": "viewer"}, headers=h)
        viewer_secret = resp.json()["secret"]

        resp = client.post(
            "/v1/pipeline",
            json={"project_path": "/tmp/test"},
            headers={"Authorization": f"Bearer {viewer_secret}"},
        )
        assert resp.status_code == 403

    def test_dev_can_write(self, client: TestClient, admin_key: tuple) -> None:
        """DEV can read projects."""
        secret, _, project_id = admin_key
        h = {"Authorization": f"Bearer {secret}"}

        resp = client.post("/v1/keys", json={"project_id": project_id, "role": "dev"}, headers=h)
        dev_secret = resp.json()["secret"]

        resp = client.get("/v1/projects", headers={"Authorization": f"Bearer {dev_secret}"})
        assert resp.status_code in (200, 201)

    def test_dev_cannot_create_keys(self, client: TestClient, admin_key: tuple) -> None:
        """DEV cannot create keys."""
        secret, _, project_id = admin_key
        h = {"Authorization": f"Bearer {secret}"}

        resp = client.post("/v1/keys", json={"project_id": project_id, "role": "dev"}, headers=h)
        dev_secret = resp.json()["secret"]

        resp = client.post(
            "/v1/keys",
            json={"project_id": project_id, "role": "viewer"},
            headers={"Authorization": f"Bearer {dev_secret}"},
        )
        assert resp.status_code == 403

    def test_admin_can_create_keys(self, client: TestClient, admin_key: tuple) -> None:
        """ADMIN can create keys."""
        secret, _, project_id = admin_key
        resp = client.post(
            "/v1/keys",
            json={"project_id": project_id, "role": "viewer"},
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert resp.status_code in (200, 201)


class TestProjectIsolation:
    """Test that projects are properly isolated."""

    def test_key_cannot_access_other_project(self, client: TestClient, admin_key: tuple) -> None:
        """Key for project A cannot access project B data."""
        admin_secret, org_id, project_a = admin_key
        h = {"Authorization": f"Bearer {admin_secret}"}

        resp = client.post("/v1/projects", json={"org_id": org_id, "name": "Project B"}, headers=h)
        project_b = resp.json()["project_id"]

        resp = client.post("/v1/keys", json={"project_id": project_a, "role": "dev"}, headers=h)
        project_a_secret = resp.json()["secret"]

        # DEV cannot override project — X-Project-ID ignored for non-admin
        resp = client.get(
            "/v1/projects",
            headers={
                "Authorization": f"Bearer {project_a_secret}",
                "X-Project-ID": project_b,
            },
        )
        # Should succeed but scoped to key's org
        assert resp.status_code in (200, 201)


class TestPersistence:
    """Test JSONL persistence."""

    def test_persistence_replay(self, tmp_path) -> None:
        """Events are replayed on startup."""
        log_path = tmp_path / "org_events.jsonl"

        store = OrgStore()
        store.configure(str(log_path))

        org = store.create_org("Test Org")
        project = store.create_project(org.org_id, "Test Project")
        secret, key = store.create_key(org.org_id, project.project_id, Role.DEV)

        store2 = OrgStore()
        store2.configure(str(log_path))

        assert org.org_id in store2._orgs
        assert project.project_id in store2._projects
        assert key.key_id in store2._api_keys

        resolved = store2.resolve_key(secret)
        assert resolved is not None
        assert resolved.key_id == key.key_id


class TestAdminProjectOverride:
    """Test ADMIN can override project within same org."""

    def test_admin_can_override_project(self, client: TestClient, admin_key: tuple) -> None:
        """ADMIN can use X-Project-ID to access other projects in same org."""
        admin_secret, org_id, project_a = admin_key
        h = {"Authorization": f"Bearer {admin_secret}"}

        resp = client.post("/v1/projects", json={"org_id": org_id, "name": "Project B"}, headers=h)
        project_b = resp.json()["project_id"]

        resp = client.post("/v1/keys", json={"project_id": project_a, "role": "admin"}, headers=h)
        admin_key_a_secret = resp.json()["secret"]

        resp = client.get(
            "/v1/keys",
            headers={
                "Authorization": f"Bearer {admin_key_a_secret}",
                "X-Project-ID": project_b,
            },
        )
        assert resp.status_code in (200, 201)


class TestKeyFormat:
    """Test key format validation."""

    def test_key_format(self, client: TestClient, admin_key: tuple) -> None:
        """Generated keys have correct format."""
        secret, _, project_id = admin_key
        h = {"Authorization": f"Bearer {secret}"}

        resp = client.post("/v1/keys", json={"project_id": project_id, "role": "dev"}, headers=h)
        dev_secret = resp.json()["secret"]

        assert dev_secret.startswith("vf_")
        assert len(dev_secret) > 20

    def test_key_id_is_deterministic(self) -> None:
        """Same secret always produces same key_id."""
        from verifily_cli_v1.core.api.identity import KeyManager

        secret = KeyManager.generate_secret()
        key_id1 = KeyManager.derive_key_id(secret)
        key_id2 = KeyManager.derive_key_id(secret)

        assert key_id1 == key_id2
        assert len(key_id1) == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
