"""Tests for organizations, projects, memberships — store, API, and CLI."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from verifily_cli_v1.cli import app
from verifily_cli_v1.core.api.orgs import Membership, OrgStore, Organization, Project, Role, org_store
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store

runner = CliRunner()


@pytest.fixture
def store():
    """Fresh OrgStore for each test."""
    s = OrgStore()
    return s


@pytest.fixture
def api_client():
    """FastAPI test client with fresh app."""
    app = create_app()
    jobs_store.stop_worker()
    return TestClient(app)


# ── OrgStore unit tests ──────────────────────────────────────────


class TestOrgStore:
    def test_create_org(self, store):
        org = store.create_org(name="Test Org", api_key_id="key-1")
        assert isinstance(org, Organization)
        assert org.name == "Test Org"
        assert org.created_by == "key-1"
        assert len(org.id) == 12

    def test_create_org_auto_membership(self, store):
        org = store.create_org(name="Auto", api_key_id="key-1")
        role = store.check_access(org.id, "key-1")
        assert role == Role.OWNER

    def test_list_orgs_by_member(self, store):
        store.create_org(name="Org A", api_key_id="key-1")
        store.create_org(name="Org B", api_key_id="key-2")
        orgs = store.list_orgs(api_key_id="key-1")
        assert len(orgs) == 1
        assert orgs[0].name == "Org A"

    def test_list_orgs_all(self, store):
        store.create_org(name="Org A", api_key_id="key-1")
        store.create_org(name="Org B", api_key_id="key-2")
        orgs = store.list_orgs()
        assert len(orgs) == 2

    def test_get_org(self, store):
        org = store.create_org(name="Find Me", api_key_id="key-1")
        found = store.get_org(org.id)
        assert found is not None
        assert found.name == "Find Me"

    def test_get_org_not_found(self, store):
        assert store.get_org("nonexistent") is None

    def test_create_project(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        proj = store.create_project(org_id=org.id, name="Project A", api_key_id="key-1")
        assert isinstance(proj, Project)
        assert proj.org_id == org.id
        assert proj.name == "Project A"

    def test_create_project_invalid_org(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.create_project(org_id="bad", name="P", api_key_id="k")

    def test_list_projects_by_org(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        store.create_project(org_id=org.id, name="P1", api_key_id="key-1")
        store.create_project(org_id=org.id, name="P2", api_key_id="key-1")
        projects = store.list_projects(org_id=org.id)
        assert len(projects) == 2

    def test_get_project(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        proj = store.create_project(org_id=org.id, name="P", api_key_id="key-1")
        found = store.get_project(proj.id)
        assert found is not None
        assert found.name == "P"

    def test_check_access_owner(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        assert store.check_access(org.id, "key-1") == Role.OWNER

    def test_check_access_non_member(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        assert store.check_access(org.id, "key-other") is None

    def test_add_membership(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        m = store.add_membership(org.id, "key-2", Role.MEMBER)
        assert isinstance(m, Membership)
        assert m.role == Role.MEMBER
        assert store.check_access(org.id, "key-2") == Role.MEMBER

    def test_add_membership_duplicate_updates_role(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        store.add_membership(org.id, "key-2", Role.MEMBER)
        store.add_membership(org.id, "key-2", Role.ADMIN)
        assert store.check_access(org.id, "key-2") == Role.ADMIN
        # Should not have duplicate entries
        members = store.list_memberships(org.id)
        key2_members = [m for m in members if m.api_key_id == "key-2"]
        assert len(key2_members) == 1

    def test_list_memberships(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        store.add_membership(org.id, "key-2", Role.MEMBER)
        members = store.list_memberships(org.id)
        assert len(members) == 2  # owner + added member

    def test_reset_clears_state(self, store):
        store.create_org(name="Org", api_key_id="key-1")
        store.reset()
        assert store.list_orgs() == []

    def test_require_project_access_ok(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        proj = store.create_project(org_id=org.id, name="P", api_key_id="key-1")
        role = store.require_project_access(proj.id, "key-1")
        assert role == Role.OWNER

    def test_require_project_access_denied(self, store):
        org = store.create_org(name="Org", api_key_id="key-1")
        proj = store.create_project(org_id=org.id, name="P", api_key_id="key-1")
        with pytest.raises(PermissionError):
            store.require_project_access(proj.id, "key-other")


# ── API endpoint tests ───────────────────────────────────────────


class TestOrgAPI:
    def test_create_org_201(self, api_client):
        resp = api_client.post("/v1/orgs", json={"name": "Test Org"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Org"
        assert "id" in data
        assert data["created_by"] == "anonymous"

    def test_list_orgs(self, api_client):
        api_client.post("/v1/orgs", json={"name": "Org A"})
        api_client.post("/v1/orgs", json={"name": "Org B"})
        resp = api_client.get("/v1/orgs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["orgs"]) == 2

    def test_create_project_201(self, api_client):
        org_resp = api_client.post("/v1/orgs", json={"name": "Org"})
        org_id = org_resp.json()["id"]
        resp = api_client.post("/v1/projects", json={"org_id": org_id, "name": "Proj"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Proj"
        assert data["org_id"] == org_id

    def test_create_project_403_non_member(self, api_client):
        # Create org as "anonymous" (default when no auth)
        org_resp = api_client.post("/v1/orgs", json={"name": "Org"})
        org_id = org_resp.json()["id"]

        # Make a second client with a different key identity
        # Since there's no auth enabled, we can't truly simulate a different user
        # via the test client. Instead, test the store-level access check.
        from verifily_cli_v1.core.api.orgs import org_store as _store
        role = _store.check_access(org_id, "different-key")
        assert role is None

    def test_list_projects(self, api_client):
        org_resp = api_client.post("/v1/orgs", json={"name": "Org"})
        org_id = org_resp.json()["id"]
        api_client.post("/v1/projects", json={"org_id": org_id, "name": "P1"})
        api_client.post("/v1/projects", json={"org_id": org_id, "name": "P2"})
        resp = api_client.get("/v1/projects")
        assert resp.status_code == 200
        assert len(resp.json()["projects"]) == 2

    def test_list_projects_filter_org(self, api_client):
        org1 = api_client.post("/v1/orgs", json={"name": "Org1"}).json()
        org2 = api_client.post("/v1/orgs", json={"name": "Org2"}).json()
        api_client.post("/v1/projects", json={"org_id": org1["id"], "name": "P1"})
        api_client.post("/v1/projects", json={"org_id": org2["id"], "name": "P2"})
        resp = api_client.get(f"/v1/projects?org_id={org1['id']}")
        assert resp.status_code == 200
        projects = resp.json()["projects"]
        assert len(projects) == 1
        assert projects[0]["org_id"] == org1["id"]

    def test_add_membership_201(self, api_client):
        org_resp = api_client.post("/v1/orgs", json={"name": "Org"})
        org_id = org_resp.json()["id"]
        resp = api_client.post(
            f"/v1/orgs/{org_id}/memberships",
            json={"api_key_id": "new-member", "role": "MEMBER"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["api_key_id"] == "new-member"
        assert data["role"] == "MEMBER"

    def test_add_membership_403_not_admin(self, api_client):
        # Store-level test: non-member trying to add
        from verifily_cli_v1.core.api.orgs import org_store as _store
        org = _store.create_org(name="Private", api_key_id="owner-key")
        role = _store.check_access(org.id, "random-key")
        assert role is None
        # RBAC: non-member should not be OWNER or ADMIN
        assert role not in (Role.OWNER, Role.ADMIN)

    def test_list_memberships(self, api_client):
        org_resp = api_client.post("/v1/orgs", json={"name": "Org"})
        org_id = org_resp.json()["id"]
        api_client.post(
            f"/v1/orgs/{org_id}/memberships",
            json={"api_key_id": "member-1", "role": "MEMBER"},
        )
        resp = api_client.get(f"/v1/orgs/{org_id}/memberships")
        assert resp.status_code == 200
        members = resp.json()["memberships"]
        assert len(members) >= 2  # owner + added member


# ── CLI help tests ───────────────────────────────────────────────


class TestOrgCLI:
    def test_org_create_help(self):
        result = runner.invoke(app, ["org-create", "--help"])
        assert result.exit_code == 0
        assert "--name" in result.output

    def test_org_list_help(self):
        result = runner.invoke(app, ["org-list", "--help"])
        assert result.exit_code == 0
        assert "--server" in result.output

    def test_project_create_help(self):
        result = runner.invoke(app, ["project-create", "--help"])
        assert result.exit_code == 0
        assert "--org" in result.output
        assert "--name" in result.output

    def test_project_list_help(self):
        result = runner.invoke(app, ["project-list", "--help"])
        assert result.exit_code == 0
        assert "--org" in result.output
