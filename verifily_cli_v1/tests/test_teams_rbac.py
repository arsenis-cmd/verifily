"""Tests for Teams v1: organizations, users, and RBAC."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
from verifily_cli_v1.core.teams.models import ApiKeyRecord, Membership, Org, Project, User
from verifily_cli_v1.core.teams.scopes import TEAMS_SCOPES, resolve_teams_scope
from verifily_cli_v1.core.teams.store import TeamsStore, teams_store


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"


def _make_client(monkeypatch, **kwargs):
    """Create a TestClient with given settings overrides."""
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_TEAMS_ENABLED", raising=False)
    monkeypatch.delenv("VERIFILY_SUPER_ADMIN_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_TEAMS_PERSIST", raising=False)
    monkeypatch.delenv("VERIFILY_ENABLE_ADMIN", raising=False)
    settings = load_settings(**kwargs)
    app = create_app(settings)
    jobs_store.stop_worker()
    return TestClient(app)


def _ingest_drill():
    """Ingest the demo support_tickets.csv, return dataset path."""
    from verifily_cli_v1.commands.ingest import run as ingest_run

    out_dir = tempfile.mkdtemp(prefix="teams_test_")
    ingest_run(
        input_path=str(DRILL_DIR / "raw" / "support_tickets.csv"),
        output_path=out_dir,
        schema="sft",
        map_args=["question:subject", "answer:resolution", "context:body"],
        tag_args=["source:test"],
        id_col=None,
        limit=None,
        strict=False,
        dry_run=False,
        verbose=False,
    )
    return str(Path(out_dir) / "dataset.jsonl")


# ── Teams Models Tests ────────────────────────────────────────


class TestTeamsModels:
    def test_org_roundtrip(self):
        org = Org(id="org-1", name="Acme", created_at=1000.0)
        d = org.to_dict()
        assert d["id"] == "org-1"
        assert d["name"] == "Acme"
        assert d["created_at"] == 1000.0

    def test_user_roundtrip(self):
        user = User(id="user-1", email="a@b.com", name="Alice", created_at=1000.0)
        d = user.to_dict()
        assert d["id"] == "user-1"
        assert d["email"] == "a@b.com"
        assert d["disabled"] is False

    def test_membership_roundtrip(self):
        mem = Membership(user_id="u1", org_id="o1", role="admin", created_at=1000.0)
        d = mem.to_dict()
        assert d["user_id"] == "u1"
        assert d["role"] == "admin"

    def test_project_roundtrip(self):
        proj = Project(id="p1", org_id="o1", name="Prod", created_at=1000.0)
        d = proj.to_dict()
        assert d["id"] == "p1"
        assert d["org_id"] == "o1"

    def test_api_key_roundtrip(self):
        rec = ApiKeyRecord(
            id="k1", org_id="o1", name="Key 1", key_hash="abc",
            scopes=["run:write"], project_ids=["p1"],
            created_at=1000.0, created_by="u1",
        )
        d = rec.to_dict()
        assert d["id"] == "k1"
        assert d["scopes"] == ["run:write"]
        assert "key_hash" not in d  # hash should not be in public dict


# ── Teams Store Tests ─────────────────────────────────────────


class TestTeamsStore:
    def test_create_org(self):
        store = TeamsStore()
        org = store.create_org(id="org-1", name="Acme")
        assert org.id == "org-1"
        assert store.get_org("org-1") is not None

    def test_create_user(self):
        store = TeamsStore()
        user = store.create_user(id="u1", email="a@b.com", name="Alice")
        assert user.id == "u1"
        assert store.get_user("u1") is not None

    def test_add_membership(self):
        store = TeamsStore()
        mem = store.add_membership(user_id="u1", org_id="o1", role="admin")
        assert mem.role == "admin"
        mems = store.get_memberships(org_id="o1")
        assert len(mems) == 1

    def test_invalid_role_raises(self):
        store = TeamsStore()
        with pytest.raises(ValueError, match="Invalid role"):
            store.add_membership(user_id="u1", org_id="o1", role="superuser")

    def test_create_project(self):
        store = TeamsStore()
        proj = store.create_project(id="p1", org_id="o1", name="Prod")
        assert proj.id == "p1"
        assert store.get_project("p1") is not None

    def test_create_api_key_and_resolve(self):
        store = TeamsStore()
        store.configure_salt("test-salt")
        rec = store.create_api_key(
            id="k1", org_id="o1", name="Key 1", raw_key="sk-test-secret-key",
            scopes=["run:write"], project_ids=["p1"], created_by="u1",
        )
        assert rec.id == "k1"

        # Resolve by raw key
        found = store.resolve_key("sk-test-secret-key")
        assert found is not None
        assert found.id == "k1"

    def test_resolve_unknown_key_none(self):
        store = TeamsStore()
        store.configure_salt("test-salt")
        assert store.resolve_key("sk-unknown") is None

    def test_persistence_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        store1 = TeamsStore()
        store1.configure_salt("salt")
        store1.configure_persistence(path)
        store1.create_org(id="o1", name="Acme")
        store1.create_user(id="u1", email="a@b.com", name="Alice")
        store1.add_membership(user_id="u1", org_id="o1", role="owner")
        store1.create_project(id="p1", org_id="o1", name="Prod")
        store1.create_api_key(
            id="k1", org_id="o1", name="Key 1", raw_key="sk-test-persist",
            scopes=["run:write"], project_ids=["p1"], created_by="u1",
        )

        # New store replays from file
        store2 = TeamsStore()
        store2.configure_salt("salt")
        store2.configure_persistence(path)
        assert store2.get_org("o1") is not None
        assert store2.get_user("u1") is not None
        assert len(store2.get_memberships(org_id="o1")) == 1
        assert store2.get_project("p1") is not None
        # Note: resolve_key won't work for replayed keys since hash
        # was computed by store1 with same salt - should still match
        assert store2.resolve_key("sk-test-persist") is not None

    def test_reset_clears(self):
        store = TeamsStore()
        store.create_org(id="o1", name="Acme")
        store.reset()
        assert store.get_org("o1") is None


# ── Teams Scopes Tests ────────────────────────────────────────


class TestTeamsScopes:
    def test_four_scopes_exist(self):
        assert len(TEAMS_SCOPES) == 4
        assert "run:write" in TEAMS_SCOPES
        assert "run:read" in TEAMS_SCOPES
        assert "usage:read" in TEAMS_SCOPES
        assert "admin:write" in TEAMS_SCOPES

    def test_resolve_pipeline_post(self):
        assert resolve_teams_scope("POST", "/v1/pipeline") == "run:write"

    def test_resolve_report_post(self):
        assert resolve_teams_scope("POST", "/v1/report") == "run:write"

    def test_resolve_jobs_get(self):
        assert resolve_teams_scope("GET", "/v1/jobs/abc123") == "run:read"

    def test_resolve_usage_get(self):
        assert resolve_teams_scope("GET", "/v1/usage") == "usage:read"

    def test_resolve_billing_get(self):
        assert resolve_teams_scope("GET", "/v1/billing/events") == "usage:read"

    def test_resolve_unknown_path_none(self):
        assert resolve_teams_scope("GET", "/health") is None

    def test_resolve_monitor_post(self):
        assert resolve_teams_scope("POST", "/v1/monitor/start") == "run:write"


# ── Teams API Tests ───────────────────────────────────────────


class TestTeamsAPI:
    def test_teams_disabled_404(self, monkeypatch):
        client = _make_client(monkeypatch, teams_enabled=False)
        resp = client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"})
        assert resp.status_code == 404 or resp.status_code == 405

    def test_admin_requires_super_key(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        # No key -> 401
        resp = client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"})
        assert resp.status_code == 401

        # Wrong key -> 401
        resp = client.post(
            "/v1/admin/orgs",
            json={"id": "o1", "name": "Acme"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_admin_create_org(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        resp = client.post(
            "/v1/admin/orgs",
            json={"id": "org-acme", "name": "Acme Corp"},
            headers={"Authorization": "Bearer sk-super-secret"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "org-acme"
        assert data["name"] == "Acme Corp"

    def test_admin_create_user(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        resp = client.post(
            "/v1/admin/users",
            json={"id": "user-alice", "email": "alice@acme.com", "name": "Alice"},
            headers={"Authorization": "Bearer sk-super-secret"},
        )
        assert resp.status_code == 201
        assert resp.json()["id"] == "user-alice"

    def test_admin_add_membership(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        headers = {"Authorization": "Bearer sk-super-secret"}
        client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"}, headers=headers)
        client.post("/v1/admin/users", json={"id": "u1", "email": "a@b.com", "name": "A"}, headers=headers)

        resp = client.post(
            "/v1/admin/memberships",
            json={"user_id": "u1", "org_id": "o1", "role": "admin"},
            headers=headers,
        )
        assert resp.status_code == 201
        assert resp.json()["role"] == "admin"

    def test_admin_create_project(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        headers = {"Authorization": "Bearer sk-super-secret"}
        client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"}, headers=headers)

        resp = client.post(
            "/v1/admin/team-projects",
            json={"id": "proj-prod", "org_id": "o1", "name": "Production"},
            headers=headers,
        )
        assert resp.status_code == 201
        assert resp.json()["org_id"] == "o1"

    def test_admin_list_projects(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        headers = {"Authorization": "Bearer sk-super-secret"}
        client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"}, headers=headers)
        client.post(
            "/v1/admin/team-projects",
            json={"id": "p1", "org_id": "o1", "name": "P1"},
            headers=headers,
        )
        resp = client.get("/v1/admin/team-projects", headers=headers)
        assert resp.status_code == 200
        assert len(resp.json()["projects"]) == 1

    def test_admin_issue_key_and_whoami(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        admin_headers = {"Authorization": "Bearer sk-super-secret"}
        client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"}, headers=admin_headers)
        client.post("/v1/admin/users", json={"id": "u1", "email": "a@b.com", "name": "A"}, headers=admin_headers)
        client.post(
            "/v1/admin/team-projects",
            json={"id": "p1", "org_id": "o1", "name": "Prod"},
            headers=admin_headers,
        )

        resp = client.post(
            "/v1/admin/api-keys",
            json={
                "id": "k1", "org_id": "o1", "name": "Key 1",
                "raw_key": "sk-test-team-key-12345",
                "scopes": ["run:write", "run:read"],
                "project_ids": ["p1"],
                "created_by": "u1",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["id"] == "k1"

        # Whoami with super admin key
        whoami_resp = client.get("/v1/admin/whoami", headers=admin_headers)
        assert whoami_resp.status_code == 200
        assert whoami_resp.json()["api_key_id"] == "super-admin"

    def test_scope_enforcement(self, monkeypatch):
        """Key without run:write cannot POST /v1/report."""
        ds = _ingest_drill()
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        admin_headers = {"Authorization": "Bearer sk-super-secret"}
        client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"}, headers=admin_headers)
        client.post(
            "/v1/admin/team-projects",
            json={"id": "p1", "org_id": "o1", "name": "Prod"},
            headers=admin_headers,
        )

        # Issue key with only run:read (no run:write)
        client.post(
            "/v1/admin/api-keys",
            json={
                "id": "k-readonly", "org_id": "o1", "name": "ReadOnly",
                "raw_key": "sk-readonly-key-12345678",
                "scopes": ["run:read"],
                "project_ids": ["*"],
                "created_by": "admin",
            },
            headers=admin_headers,
        )

        # Try to POST /v1/report with read-only key -> 403
        resp = client.post(
            "/v1/report",
            json={"dataset_path": ds, "schema": "sft"},
            headers={"Authorization": "Bearer sk-readonly-key-12345678"},
        )
        assert resp.status_code == 403

    def test_project_enforcement(self, monkeypatch):
        """Key bound to proj-A cannot access proj-B."""
        ds = _ingest_drill()
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        admin_headers = {"Authorization": "Bearer sk-super-secret"}
        client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"}, headers=admin_headers)
        client.post(
            "/v1/admin/team-projects",
            json={"id": "proj-a", "org_id": "o1", "name": "Project A"},
            headers=admin_headers,
        )

        # Issue key bound to proj-a only
        client.post(
            "/v1/admin/api-keys",
            json={
                "id": "k-proj-a", "org_id": "o1", "name": "ProjA Key",
                "raw_key": "sk-proj-a-key-12345678",
                "scopes": ["run:write", "run:read"],
                "project_ids": ["proj-a"],
                "created_by": "admin",
            },
            headers=admin_headers,
        )

        key_headers = {"Authorization": "Bearer sk-proj-a-key-12345678"}

        # Access with correct project -> 200
        resp = client.post(
            "/v1/report",
            json={"dataset_path": ds, "schema": "sft"},
            headers={**key_headers, "X-Project-ID": "proj-a"},
        )
        assert resp.status_code == 200

        # Access with wrong project -> 403
        resp = client.post(
            "/v1/report",
            json={"dataset_path": ds, "schema": "sft"},
            headers={**key_headers, "X-Project-ID": "proj-b"},
        )
        assert resp.status_code == 403

    def test_wildcard_project_access(self, monkeypatch):
        """Key with project_ids=["*"] can access any project."""
        ds = _ingest_drill()
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        admin_headers = {"Authorization": "Bearer sk-super-secret"}
        client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"}, headers=admin_headers)

        # Issue wildcard key
        client.post(
            "/v1/admin/api-keys",
            json={
                "id": "k-wildcard", "org_id": "o1", "name": "Wildcard",
                "raw_key": "sk-wildcard-key-12345678",
                "scopes": ["run:write", "run:read"],
                "project_ids": ["*"],
                "created_by": "admin",
            },
            headers=admin_headers,
        )

        key_headers = {"Authorization": "Bearer sk-wildcard-key-12345678"}

        resp = client.post(
            "/v1/report",
            json={"dataset_path": ds, "schema": "sft"},
            headers={**key_headers, "X-Project-ID": "any-project"},
        )
        assert resp.status_code == 200

    def test_no_key_401(self, monkeypatch):
        """Request without key -> 401."""
        ds = _ingest_drill()
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        resp = client.post(
            "/v1/report",
            json={"dataset_path": ds, "schema": "sft"},
        )
        assert resp.status_code == 401

    def test_invalid_scope_rejected(self, monkeypatch):
        """Issuing key with invalid scope returns 422."""
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="sk-super-secret",
            key_salt="test-salt",
        )
        admin_headers = {"Authorization": "Bearer sk-super-secret"}
        client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"}, headers=admin_headers)

        resp = client.post(
            "/v1/admin/api-keys",
            json={
                "id": "k-bad", "org_id": "o1", "name": "Bad",
                "raw_key": "sk-bad-key-123456789",
                "scopes": ["run:write", "bogus:scope"],
                "project_ids": ["*"],
                "created_by": "admin",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 422

    def test_bootstrap_mode_no_super_key(self, monkeypatch):
        """When super_admin_key is empty, admin endpoints are open (bootstrap)."""
        client = _make_client(
            monkeypatch,
            teams_enabled=True,
            auth_mode="teams",
            super_admin_key="",
            key_salt="test-salt",
        )
        resp = client.post("/v1/admin/orgs", json={"id": "o1", "name": "Acme"})
        assert resp.status_code == 201
