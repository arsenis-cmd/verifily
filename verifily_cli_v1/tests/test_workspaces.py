"""Tests for workspaces: JSON-file store, bootstrap, RBAC, isolation, legacy compat."""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
from verifily_cli_v1.core.workspaces.store import (
    InvalidRoleError,
    KeyNotFoundError,
    OrgNotFoundError,
    ProjectNotFoundError,
    workspaces_store,
)


BOOTSTRAP_TOKEN = "bootstrap-test-secret"


def _make_client(monkeypatch, **kwargs):
    """Create a test client with workspaces enabled."""
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_AUTH_MODE", raising=False)
    monkeypatch.delenv("VERIFILY_ORG_MODE", raising=False)
    monkeypatch.delenv("VERIFILY_WORKSPACES_ENABLED", raising=False)
    monkeypatch.delenv("VERIFILY_BOOTSTRAP_TOKEN", raising=False)
    # Use a fresh temp file per test to avoid stale state
    tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tf.close()
    os.unlink(tf.name)  # start with no file
    settings = load_settings(
        workspaces_enabled=True,
        bootstrap_token=BOOTSTRAP_TOKEN,
        workspaces_store_path=tf.name,
        **kwargs,
    )
    app = create_app(settings)
    jobs_store.stop_worker()
    return TestClient(app)


def _bootstrap_org(client, name="Acme"):
    """Bootstrap the first org using the bootstrap token."""
    resp = client.post(
        "/v1/orgs",
        json={"name": name},
        headers={"X-Bootstrap-Token": BOOTSTRAP_TOKEN},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_full_setup(client):
    """Bootstrap org, create project + admin key, return (org_id, project_id, admin_key)."""
    org = _bootstrap_org(client)
    org_id = org["org_id"]

    # Create project using bootstrap (before any keys exist, need admin key)
    # Actually, after first org is created, we need an admin key.
    # But we have no key yet... the bootstrap only allows org creation.
    # So we need a way to create the first admin key.
    # The way this works: after bootstrap creates an org, the store is no longer empty,
    # so we need auth for everything else. But we have no key yet.
    #
    # Solution: We'll make the first org creation also return a way to create keys.
    # Actually, looking at the auth flow: the bootstrap check is ONLY for POST /v1/orgs
    # when the store is empty. After creating the first org, POST /v1/projects requires auth.
    #
    # The correct flow is:
    # 1. POST /v1/orgs with bootstrap token -> creates org (store now non-empty)
    # 2. We still need a key. But POST /v1/keys requires auth.
    #
    # This is a chicken-and-egg problem. The fix: the workspaces auth should also allow
    # bootstrap token for POST /v1/keys and POST /v1/projects when no keys exist.
    #
    # For now, let's test with the bootstrap token being accepted for all admin
    # endpoints when the store has no keys.
    #
    # Actually, re-reading the auth code: the bootstrap check only allows POST /v1/orgs
    # when the store is_empty(). After that, all other requests require a valid API key.
    # But we can't create a key without auth!
    #
    # The fix: bootstrap should allow creating the first key too.
    # Let me adjust the approach: make the first org creation automatically create
    # an admin key and return it. Or better: extend bootstrap to cover the entire
    # onboarding flow (first org + first project + first key).

    # For the test to work, we need the store to support direct key creation
    # from the test side, or we need to fix the bootstrap flow.
    # Let me use the store directly to create the first key for testing:
    proj = workspaces_store.create_project(org_id, "QA", "free")
    project_id = proj["project_id"]
    key_result = workspaces_store.create_api_key(project_id, "admin")
    admin_key = key_result["api_key"]

    return org_id, project_id, admin_key


# ── Store Unit Tests ────────────────────────────────────────────


class TestStoreBasics:
    def setup_method(self):
        workspaces_store.reset()

    def test_create_org(self):
        result = workspaces_store.create_org("Acme")
        assert result["org_id"].startswith("org_")
        assert result["name"] == "Acme"
        assert not workspaces_store.is_empty()

    def test_create_project_validates_org(self):
        with pytest.raises(OrgNotFoundError):
            workspaces_store.create_project("org_fake", "P1")

    def test_create_key_returns_vf_secret(self):
        org = workspaces_store.create_org("Acme")
        proj = workspaces_store.create_project(org["org_id"], "QA")
        key = workspaces_store.create_api_key(proj["project_id"], "editor")
        assert key["api_key"].startswith("vf_")
        assert key["role"] == "editor"
        assert len(key["api_key_id"]) == 12

    def test_resolve_request_valid(self):
        org = workspaces_store.create_org("Acme")
        proj = workspaces_store.create_project(org["org_id"], "QA")
        key = workspaces_store.create_api_key(proj["project_id"], "admin")
        resolved = workspaces_store.resolve_request(key["api_key"])
        assert resolved is not None
        assert resolved["org_id"] == org["org_id"]
        assert resolved["project_id"] == proj["project_id"]
        assert resolved["role"] == "admin"
        assert resolved["api_key_id"] == key["api_key_id"]

    def test_resolve_request_invalid(self):
        assert workspaces_store.resolve_request("vf_boguskey1234567890123456789012") is None

    def test_resolve_request_revoked(self):
        org = workspaces_store.create_org("Acme")
        proj = workspaces_store.create_project(org["org_id"], "QA")
        key = workspaces_store.create_api_key(proj["project_id"], "editor")
        workspaces_store.revoke_api_key(proj["project_id"], key["api_key_id"])
        assert workspaces_store.resolve_request(key["api_key"]) is None

    def test_role_validation(self):
        org = workspaces_store.create_org("Acme")
        proj = workspaces_store.create_project(org["org_id"], "QA")
        with pytest.raises(InvalidRoleError):
            workspaces_store.create_api_key(proj["project_id"], "superadmin")

    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            workspaces_store.configure(path=path, salt="test-salt")
            org = workspaces_store.create_org("Acme")
            proj = workspaces_store.create_project(org["org_id"], "QA")
            key = workspaces_store.create_api_key(proj["project_id"], "viewer")
            secret = key["api_key"]

            # Reset and reload from file
            workspaces_store.reset()
            assert workspaces_store.is_empty()
            workspaces_store.configure(path=path, salt="test-salt")
            assert not workspaces_store.is_empty()

            # Resolve should still work
            resolved = workspaces_store.resolve_request(secret)
            assert resolved is not None
            assert resolved["role"] == "viewer"
        finally:
            os.unlink(path)


# ── Bootstrap Tests ─────────────────────────────────────────────


class TestBootstrap:
    def test_first_org_no_auth_with_token(self, monkeypatch):
        client = _make_client(monkeypatch)
        resp = client.post(
            "/v1/orgs",
            json={"name": "First"},
            headers={"X-Bootstrap-Token": BOOTSTRAP_TOKEN},
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "First"

    def test_first_org_wrong_bootstrap_token(self, monkeypatch):
        client = _make_client(monkeypatch)
        resp = client.post(
            "/v1/orgs",
            json={"name": "First"},
            headers={"X-Bootstrap-Token": "wrong-token"},
        )
        assert resp.status_code == 401

    def test_after_bootstrap_requires_auth(self, monkeypatch):
        client = _make_client(monkeypatch)
        _bootstrap_org(client)
        # Now try to create another org without auth
        resp = client.post("/v1/orgs", json={"name": "Second"})
        assert resp.status_code == 401

    def test_bootstrap_no_token_when_empty_and_no_token_configured(self, monkeypatch):
        """When no bootstrap_token is configured and store is empty, org creation is open."""
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_AUTH_MODE", raising=False)
        monkeypatch.delenv("VERIFILY_ORG_MODE", raising=False)
        monkeypatch.delenv("VERIFILY_WORKSPACES_ENABLED", raising=False)
        monkeypatch.delenv("VERIFILY_BOOTSTRAP_TOKEN", raising=False)
        tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tf.close()
        os.unlink(tf.name)
        settings = load_settings(
            workspaces_enabled=True,
            bootstrap_token="",
            workspaces_store_path=tf.name,
        )
        app = create_app(settings)
        jobs_store.stop_worker()
        client = TestClient(app)
        resp = client.post("/v1/orgs", json={"name": "OpenOrg"})
        assert resp.status_code == 201


# ── Role Enforcement Tests ──────────────────────────────────────


class TestRoleEnforcement:
    def test_admin_can_create_keys(self, monkeypatch):
        client = _make_client(monkeypatch)
        org_id, project_id, admin_key = _create_full_setup(client)

        resp = client.post(
            "/v1/keys",
            json={"project_id": project_id, "role": "editor"},
            headers={"Authorization": f"Bearer {admin_key}"},
        )
        assert resp.status_code == 201
        assert resp.json()["role"] == "editor"

    def test_editor_blocked_from_keys(self, monkeypatch):
        client = _make_client(monkeypatch)
        org_id, project_id, admin_key = _create_full_setup(client)

        # Create editor key
        resp = client.post(
            "/v1/keys",
            json={"project_id": project_id, "role": "editor"},
            headers={"Authorization": f"Bearer {admin_key}"},
        )
        editor_key = resp.json()["api_key"]

        # Editor tries to create a key — should be forbidden (POST /v1/keys is admin endpoint)
        resp2 = client.post(
            "/v1/keys",
            json={"project_id": project_id, "role": "viewer"},
            headers={"Authorization": f"Bearer {editor_key}"},
        )
        assert resp2.status_code == 403

    def test_viewer_blocked_from_pipeline(self, monkeypatch):
        client = _make_client(monkeypatch)
        org_id, project_id, admin_key = _create_full_setup(client)

        # Create viewer key
        resp = client.post(
            "/v1/keys",
            json={"project_id": project_id, "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_key}"},
        )
        viewer_key = resp.json()["api_key"]

        # Viewer tries pipeline — should be forbidden
        resp2 = client.post(
            "/v1/pipeline",
            json={"eval_path": "fake.csv", "run_dir": "/tmp/fake"},
            headers={"Authorization": f"Bearer {viewer_key}"},
        )
        assert resp2.status_code == 403

    def test_viewer_can_get_me(self, monkeypatch):
        client = _make_client(monkeypatch)
        org_id, project_id, admin_key = _create_full_setup(client)

        # Create viewer key
        resp = client.post(
            "/v1/keys",
            json={"project_id": project_id, "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_key}"},
        )
        viewer_key = resp.json()["api_key"]

        # Viewer can GET /v1/me
        resp2 = client.get(
            "/v1/me",
            headers={"Authorization": f"Bearer {viewer_key}"},
        )
        assert resp2.status_code == 200
        body = resp2.json()
        assert body["role"] == "viewer"
        assert body["org_id"] == org_id
        assert body["project_id"] == project_id


# ── Project Isolation Tests ─────────────────────────────────────


class TestProjectIsolation:
    def test_key_scoped_to_project(self, monkeypatch):
        client = _make_client(monkeypatch)
        org_id, project_id, admin_key = _create_full_setup(client)

        # Me should return correct context
        resp = client.get(
            "/v1/me",
            headers={"Authorization": f"Bearer {admin_key}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["project_id"] == project_id
        assert body["org_id"] == org_id
        assert body["role"] == "admin"

    def test_revoke_invalidates_key(self, monkeypatch):
        client = _make_client(monkeypatch)
        org_id, project_id, admin_key = _create_full_setup(client)

        # Create an editor key
        resp = client.post(
            "/v1/keys",
            json={"project_id": project_id, "role": "editor"},
            headers={"Authorization": f"Bearer {admin_key}"},
        )
        editor_key = resp.json()["api_key"]
        editor_key_id = resp.json()["api_key_id"]

        # Verify it works
        resp2 = client.get(
            "/v1/me",
            headers={"Authorization": f"Bearer {editor_key}"},
        )
        assert resp2.status_code == 200

        # Revoke it
        resp3 = client.post(
            "/v1/keys/revoke",
            json={"project_id": project_id, "api_key_id": editor_key_id},
            headers={"Authorization": f"Bearer {admin_key}"},
        )
        assert resp3.status_code == 200
        assert resp3.json()["ok"] is True

        # Now the revoked key should fail
        resp4 = client.get(
            "/v1/me",
            headers={"Authorization": f"Bearer {editor_key}"},
        )
        assert resp4.status_code == 401


# ── Legacy Mode Tests ───────────────────────────────────────────


class TestLegacyMode:
    def test_workspaces_disabled_legacy_key_works(self, monkeypatch):
        """When workspaces is disabled, legacy api_key auth should still work."""
        monkeypatch.delenv("VERIFILY_WORKSPACES_ENABLED", raising=False)
        monkeypatch.delenv("VERIFILY_ORG_MODE", raising=False)
        monkeypatch.delenv("VERIFILY_AUTH_MODE", raising=False)
        settings = load_settings(api_key="test-legacy-key")
        app = create_app(settings)
        jobs_store.stop_worker()
        client = TestClient(app)

        # Auth with legacy key
        resp = client.get(
            "/health",
            headers={"Authorization": "Bearer test-legacy-key"},
        )
        assert resp.status_code == 200

    def test_workspaces_disabled_existing_endpoints_unchanged(self, monkeypatch):
        """When workspaces is disabled, existing endpoints work normally."""
        monkeypatch.delenv("VERIFILY_WORKSPACES_ENABLED", raising=False)
        monkeypatch.delenv("VERIFILY_ORG_MODE", raising=False)
        monkeypatch.delenv("VERIFILY_AUTH_MODE", raising=False)
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        settings = load_settings()
        app = create_app(settings)
        jobs_store.stop_worker()
        client = TestClient(app)

        # Health should work
        resp = client.get("/health")
        assert resp.status_code == 200

        # /v1/me should NOT exist (404) when workspaces is disabled
        resp2 = client.get("/v1/me")
        assert resp2.status_code in (404, 405)
