"""Tests for Enterprise Trust v1: scoped keys, project binding, quotas, admin endpoints."""

from __future__ import annotations

import json
import tempfile

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.auth_registry import (
    AuthRegistry,
    _hash_key,
    _verify_key,
    auth_registry,
    resolve_scope,
)
from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.quotas import QuotaStore, quota_store
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings


def _make_client(monkeypatch, **kwargs):
    """Create a TestClient with given settings overrides."""
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_AUTH_MODE", raising=False)
    monkeypatch.delenv("VERIFILY_ENABLE_ADMIN", raising=False)
    monkeypatch.delenv("VERIFILY_KEY_SALT", raising=False)
    settings = load_settings(**kwargs)
    app = create_app(settings)
    jobs_store.stop_worker()
    return TestClient(app)


# ── Simple Mode Unchanged ─────────────────────────────────────


class TestSimpleModeUnchanged:
    def test_no_auth_passes(self, monkeypatch):
        client = _make_client(monkeypatch)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_simple_key_works(self, monkeypatch):
        monkeypatch.setenv("VERIFILY_API_KEY", "sk-test-key-123")
        settings = load_settings()
        app = create_app(settings)
        jobs_store.stop_worker()
        client = TestClient(app)

        resp = client.get("/v1/usage", headers={"Authorization": "Bearer sk-test-key-123"})
        assert resp.status_code == 200

    def test_simple_wrong_key_401(self, monkeypatch):
        monkeypatch.setenv("VERIFILY_API_KEY", "sk-test-key-123")
        settings = load_settings()
        app = create_app(settings)
        jobs_store.stop_worker()
        client = TestClient(app)

        resp = client.get("/v1/usage", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401


# ── Auth Registry Unit Tests ──────────────────────────────────


class TestRegistry:
    def test_create_key_stores_hash(self):
        reg = AuthRegistry(salt="test-salt")
        rec = reg.create_key(
            id="k1", name="Test", raw_key="sk-secret-abc",
            scopes=["report:read"], projects_allowed=["*"],
        )
        assert rec.key_hash != "sk-secret-abc"
        assert len(rec.key_hash) == 64  # SHA256 hex

    def test_authenticate_valid_key(self):
        reg = AuthRegistry(salt="test-salt")
        reg.create_key(
            id="k1", name="Test", raw_key="sk-secret-abc",
            scopes=["report:read"], projects_allowed=["*"],
        )
        result = reg.authenticate("sk-secret-abc")
        assert result is not None
        assert result.id == "k1"

    def test_authenticate_wrong_key(self):
        reg = AuthRegistry(salt="test-salt")
        reg.create_key(
            id="k1", name="Test", raw_key="sk-secret-abc",
            scopes=["report:read"], projects_allowed=["*"],
        )
        result = reg.authenticate("sk-wrong-key")
        assert result is None

    def test_authenticate_disabled_key_rejected(self):
        reg = AuthRegistry(salt="test-salt")
        reg.create_key(
            id="k1", name="Test", raw_key="sk-secret-abc",
            scopes=["report:read"], projects_allowed=["*"],
        )
        reg.disable_key("k1")
        result = reg.authenticate("sk-secret-abc")
        assert result is None

    def test_rotate_key(self):
        reg = AuthRegistry(salt="test-salt")
        reg.create_key(
            id="k1", name="Test", raw_key="sk-old-key",
            scopes=["report:read"], projects_allowed=["*"],
        )
        reg.rotate_key("k1", "sk-new-key")
        assert reg.authenticate("sk-old-key") is None
        assert reg.authenticate("sk-new-key") is not None

    def test_list_keys_redacted(self):
        reg = AuthRegistry(salt="test-salt")
        reg.create_key(
            id="k1", name="Test", raw_key="sk-secret-abc",
            scopes=["report:read"], projects_allowed=["*"],
        )
        keys = reg.list_keys()
        assert len(keys) == 1
        assert keys[0]["key_hash"] == "***"

    def test_persistence_replay(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        reg1 = AuthRegistry(salt="salt1")
        reg1.configure_persistence(path)
        reg1.create_project(id="p1", name="Proj 1")
        reg1.create_key(
            id="k1", name="Key 1", raw_key="sk-secret",
            scopes=["report:read"], projects_allowed=["p1"],
        )
        reg1.disable_key("k1")

        # New registry replays from the same file
        reg2 = AuthRegistry(salt="salt1")
        reg2.configure_persistence(path)
        assert reg2.get_project("p1") is not None
        assert reg2.get_project("p1").name == "Proj 1"
        keys = reg2.list_keys()
        assert len(keys) == 1
        assert keys[0]["disabled"] is True

    def test_has_admin_keys(self):
        reg = AuthRegistry(salt="test-salt")
        assert reg.has_admin_keys() is False
        reg.create_key(
            id="k1", name="Admin", raw_key="sk-admin",
            scopes=["admin:write"], projects_allowed=["*"],
        )
        assert reg.has_admin_keys() is True


# ── Scope Resolution ──────────────────────────────────────────


class TestScopeResolution:
    def test_exact_match(self):
        assert resolve_scope("/v1/report", "POST") == "report:read"
        assert resolve_scope("/v1/pipeline", "POST") == "pipeline:run"

    def test_prefix_match_jobs(self):
        assert resolve_scope("/v1/jobs/pipeline", "POST") == "jobs:submit"
        assert resolve_scope("/v1/jobs", "GET") == "jobs:read"

    def test_admin_scope(self):
        assert resolve_scope("/v1/admin/keys", "POST") == "admin:write"

    def test_public_returns_none(self):
        assert resolve_scope("/health", "GET") is None


# ── Advanced Auth ─────────────────────────────────────────────


class TestAdvancedAuth:
    def test_scope_enforcement(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        # Create key with only report:read (via bootstrap — no admin keys yet)
        resp = client.post("/v1/admin/keys", json={
            "id": "k-reader", "name": "Reader",
            "raw_key": "sk-reader-key-12345",
            "scopes": ["report:read"],
            "projects_allowed": ["*"],
        })
        assert resp.status_code == 201

        # Try to access /v1/pipeline (requires pipeline:run) with this key
        resp = client.post(
            "/v1/pipeline",
            json={"config_path": "/nonexistent"},
            headers={"Authorization": "Bearer sk-reader-key-12345"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["type"] == "AUTH_FORBIDDEN"

    def test_project_scoping(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        # Create key scoped to project "demo"
        resp = client.post("/v1/admin/keys", json={
            "id": "k-demo", "name": "Demo Key",
            "raw_key": "sk-demo-key-12345",
            "scopes": ["report:read", "pipeline:run", "usage:read"],
            "projects_allowed": ["demo"],
        })
        assert resp.status_code == 201

        # Access with correct project → should pass (may 404 on file, but not 403)
        resp = client.get(
            "/v1/usage",
            headers={
                "Authorization": "Bearer sk-demo-key-12345",
                "X-Project-ID": "demo",
            },
        )
        assert resp.status_code == 200

        # Access with wrong project → 403
        resp = client.get(
            "/v1/usage",
            headers={
                "Authorization": "Bearer sk-demo-key-12345",
                "X-Project-ID": "other",
            },
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["type"] == "AUTH_FORBIDDEN"

    def test_wildcard_project(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        resp = client.post("/v1/admin/keys", json={
            "id": "k-all", "name": "All Projects",
            "raw_key": "sk-all-key-12345678",
            "scopes": ["usage:read"],
            "projects_allowed": ["*"],
        })
        assert resp.status_code == 201

        resp = client.get(
            "/v1/usage",
            headers={
                "Authorization": "Bearer sk-all-key-12345678",
                "X-Project-ID": "anything",
            },
        )
        assert resp.status_code == 200

    def test_disabled_key_rejected(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        # Create and disable a key (bootstrap: no admin keys yet)
        client.post("/v1/admin/keys", json={
            "id": "k-temp", "name": "Temp",
            "raw_key": "sk-temp-key-12345",
            "scopes": ["usage:read"],
            "projects_allowed": ["*"],
        })
        client.post("/v1/admin/keys/{key_id}/disable".replace("{key_id}", "k-temp"), json={})

        # Now need an admin key to exist so bootstrap is closed
        client.post("/v1/admin/keys", json={
            "id": "k-admin", "name": "Admin",
            "raw_key": "sk-admin-key-12345",
            "scopes": ["admin:write", "usage:read"],
            "projects_allowed": ["*"],
        })

        # Disabled key should fail
        resp = client.get(
            "/v1/usage",
            headers={"Authorization": "Bearer sk-temp-key-12345"},
        )
        assert resp.status_code == 401


# ── Quotas ────────────────────────────────────────────────────


class TestQuotas:
    def test_quota_store_basic(self):
        qs = QuotaStore()
        qs.configure_limits(requests_per_day=2)
        assert qs.check_and_increment("k1", "p1") is None  # 1st
        assert qs.check_and_increment("k1", "p1") is None  # 2nd
        msg = qs.check_and_increment("k1", "p1")            # 3rd → exceeded
        assert msg is not None
        assert "exceeded" in msg.lower()

    def test_quota_exceeded_429(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
            quota_req_per_day=2,
        )
        # Create key
        client.post("/v1/admin/keys", json={
            "id": "k-quota", "name": "Quota Test",
            "raw_key": "sk-quota-key-12345",
            "scopes": ["usage:read"],
            "projects_allowed": ["*"],
        })
        headers = {"Authorization": "Bearer sk-quota-key-12345"}

        # First 2 requests should succeed
        assert client.get("/v1/usage", headers=headers).status_code == 200
        assert client.get("/v1/usage", headers=headers).status_code == 200

        # 3rd should hit 429
        resp = client.get("/v1/usage", headers=headers)
        assert resp.status_code == 429
        assert resp.json()["error"]["type"] == "QUOTA_EXCEEDED"
        assert "Retry-After" in resp.headers

    def test_quota_separate_projects(self):
        qs = QuotaStore()
        qs.configure_limits(requests_per_day=1)
        assert qs.check_and_increment("k1", "p1") is None
        assert qs.check_and_increment("k1", "p2") is None  # Different project
        msg = qs.check_and_increment("k1", "p1")
        assert msg is not None  # p1 is exceeded


# ── Admin Endpoints ───────────────────────────────────────────


class TestAdminEndpoints:
    def test_admin_disabled_by_default(self, monkeypatch):
        client = _make_client(monkeypatch)
        resp = client.get("/v1/admin/projects")
        assert resp.status_code in (404, 405)

    def test_admin_create_project(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        resp = client.post("/v1/admin/projects", json={"id": "proj-1", "name": "Test Project"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "proj-1"
        assert data["name"] == "Test Project"
        assert "created_at" in data

    def test_admin_list_projects(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        client.post("/v1/admin/projects", json={"id": "p1", "name": "P1"})
        client.post("/v1/admin/projects", json={"id": "p2", "name": "P2"})
        resp = client.get("/v1/admin/projects")
        assert resp.status_code == 200
        assert len(resp.json()["projects"]) == 2

    def test_admin_create_key(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        resp = client.post("/v1/admin/keys", json={
            "id": "k-new", "name": "New Key",
            "raw_key": "sk-brand-new-key-here",
            "scopes": ["report:read", "pipeline:run"],
            "projects_allowed": ["*"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "k-new"
        assert data["scopes"] == ["report:read", "pipeline:run"]
        assert data["disabled"] is False
        # key_hash should NOT appear in response
        assert "key_hash" not in data

    def test_admin_list_keys_no_hash_leak(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        client.post("/v1/admin/keys", json={
            "id": "k1", "name": "Key 1",
            "raw_key": "sk-secret-key-12345",
            "scopes": ["report:read"],
            "projects_allowed": ["*"],
        })
        resp = client.get("/v1/admin/keys")
        assert resp.status_code == 200
        keys = resp.json()["keys"]
        assert len(keys) == 1
        # key_hash should not leak through — response model omits it
        raw = json.dumps(resp.json())
        assert "sk-secret-key-12345" not in raw
        # The underlying registry redacts, but the Pydantic model filters it out entirely
        assert keys[0].get("key_hash") is None

    def test_admin_disable_key(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        client.post("/v1/admin/keys", json={
            "id": "k1", "name": "Key 1",
            "raw_key": "sk-secret-key-12345",
            "scopes": ["usage:read"],
            "projects_allowed": ["*"],
        })
        resp = client.post("/v1/admin/keys/k1/disable", json={})
        assert resp.status_code == 200
        assert resp.json()["disabled"] is True

    def test_admin_rotate_key(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        client.post("/v1/admin/keys", json={
            "id": "k1", "name": "Key 1",
            "raw_key": "sk-old-key-12345678",
            "scopes": ["usage:read"],
            "projects_allowed": ["*"],
        })
        resp = client.post("/v1/admin/keys/k1/rotate", json={"raw_key": "sk-new-key-12345678"})
        assert resp.status_code == 200
        assert resp.json()["id"] == "k1"

    def test_admin_invalid_scopes_rejected(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        resp = client.post("/v1/admin/keys", json={
            "id": "k-bad", "name": "Bad",
            "raw_key": "sk-bad-key-12345",
            "scopes": ["not:a:scope"],
            "projects_allowed": ["*"],
        })
        assert resp.status_code == 422

    def test_bootstrap_closes_after_admin_key(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced", key_salt="test-salt", enable_admin=True,
        )
        # Bootstrap: no keys → admin open
        resp = client.get("/v1/admin/projects")
        assert resp.status_code == 200

        # Create an admin key
        client.post("/v1/admin/keys", json={
            "id": "k-admin", "name": "Admin",
            "raw_key": "sk-admin-key-12345",
            "scopes": ["admin:write"],
            "projects_allowed": ["*"],
        })

        # Now unauthenticated admin should fail
        resp = client.get("/v1/admin/projects")
        assert resp.status_code == 401


# ── Hashing ───────────────────────────────────────────────────


class TestHashing:
    def test_hash_key_deterministic(self):
        h1 = _hash_key("my-key", "salt")
        h2 = _hash_key("my-key", "salt")
        assert h1 == h2

    def test_hash_key_different_salt(self):
        h1 = _hash_key("my-key", "salt1")
        h2 = _hash_key("my-key", "salt2")
        assert h1 != h2

    def test_verify_key(self):
        h = _hash_key("my-key", "salt")
        assert _verify_key("my-key", "salt", h) is True
        assert _verify_key("wrong", "salt", h) is False
