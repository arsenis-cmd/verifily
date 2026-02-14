"""SDK tests for effective config endpoint."""

from __future__ import annotations

import json
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
from verifily_sdk.models import EffectiveConfigResponse


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


class TestSDKEffectiveConfig:
    def test_effective_config_returns_dict(self, sdk_client):
        resp = sdk_client.effective_config()
        assert isinstance(resp.config, dict)

    def test_effective_config_no_api_key(self, sdk_client):
        resp = sdk_client.effective_config()
        config_str = json.dumps(resp.config)
        assert "api_key" not in config_str.lower() or "auth_enabled" in config_str

    def test_effective_config_has_privacy_mode(self, sdk_client):
        resp = sdk_client.effective_config()
        assert "privacy_mode" in resp.config

    def test_effective_config_response_model(self, sdk_client):
        resp = sdk_client.effective_config()
        assert isinstance(resp, EffectiveConfigResponse)
