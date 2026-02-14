"""SDK tests for organizations and projects."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient as StarletteTestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "verifily_sdk"))

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_sdk import VerifilyClient
from verifily_sdk.errors import ForbiddenError


def _make_sdk_client(app, api_key=None):
    """Create an SDK client backed by a starlette TestClient."""
    inner = StarletteTestClient(app, base_url="http://testserver")
    client = VerifilyClient.__new__(VerifilyClient)
    client._base_url = "http://testserver"
    client._api_key = api_key
    client._timeout = 60.0
    client._retries = 0
    client._client = inner
    return client


@pytest.fixture
def sdk_client(monkeypatch):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    app = create_app()
    jobs_store.stop_worker()
    return _make_sdk_client(app)


class TestSDKOrgs:
    def test_create_org(self, sdk_client):
        org = sdk_client.create_org(name="SDK Org")
        assert org.name == "SDK Org"
        assert len(org.id) == 12
        assert org.created_by == "anonymous"

    def test_list_orgs(self, sdk_client):
        sdk_client.create_org(name="Org A")
        sdk_client.create_org(name="Org B")
        resp = sdk_client.list_orgs()
        assert len(resp.orgs) == 2
        names = {o.name for o in resp.orgs}
        assert "Org A" in names
        assert "Org B" in names

    def test_create_project(self, sdk_client):
        org = sdk_client.create_org(name="Org")
        proj = sdk_client.create_project(org_id=org.id, name="Proj")
        assert proj.name == "Proj"
        assert proj.org_id == org.id

    def test_list_projects(self, sdk_client):
        org = sdk_client.create_org(name="Org")
        sdk_client.create_project(org_id=org.id, name="P1")
        sdk_client.create_project(org_id=org.id, name="P2")
        resp = sdk_client.list_projects(org_id=org.id)
        assert len(resp.projects) == 2

    def test_list_projects_no_filter(self, sdk_client):
        org = sdk_client.create_org(name="Org")
        sdk_client.create_project(org_id=org.id, name="P1")
        resp = sdk_client.list_projects()
        assert len(resp.projects) >= 1

    def test_forbidden_error_class(self):
        err = ForbiddenError(403, "forbidden", None, None)
        assert err.status_code == 403
        assert "forbidden" in str(err)


class TestSDKModels:
    def test_org_response_fields(self, sdk_client):
        org = sdk_client.create_org(name="Fields Test")
        assert hasattr(org, "id")
        assert hasattr(org, "name")
        assert hasattr(org, "created_at")
        assert hasattr(org, "created_by")

    def test_project_response_fields(self, sdk_client):
        org = sdk_client.create_org(name="Org")
        proj = sdk_client.create_project(org_id=org.id, name="Proj")
        assert hasattr(proj, "id")
        assert hasattr(proj, "org_id")
        assert hasattr(proj, "name")
        assert hasattr(proj, "created_at")
        assert hasattr(proj, "created_by")
