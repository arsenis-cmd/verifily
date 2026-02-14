"""Tests for hosted readiness: settings, docs gating, nonlocal blocking."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.server import create_app, validate_host
from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.settings import Settings, load_settings


# ── Settings ─────────────────────────────────────────────────────


class TestSettings:
    def test_dev_enables_docs_by_default(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_ENV", raising=False)
        monkeypatch.delenv("VERIFILY_ENABLE_DOCS", raising=False)
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        settings = load_settings(env="dev")
        assert settings.enable_docs is True

    def test_prod_disables_docs_by_default(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_ENABLE_DOCS", raising=False)
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        settings = load_settings(env="prod")
        assert settings.enable_docs is False

    def test_prod_enable_docs_override(self, monkeypatch):
        monkeypatch.setenv("VERIFILY_ENABLE_DOCS", "1")
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        settings = load_settings(env="prod")
        assert settings.enable_docs is True

    def test_settings_redacts_api_key(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        settings = load_settings(api_key="sk-secret-123")
        r = repr(settings)
        assert "sk-secret-123" not in r
        assert "***" in r

        d = settings.to_dict()
        assert d["api_key"] == "configured"
        assert "sk-secret-123" not in str(d)

    def test_settings_no_key_repr(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        settings = load_settings()
        d = settings.to_dict()
        assert d["api_key"] == "not set"


# ── Docs Gating ──────────────────────────────────────────────────


class TestDocsGating:
    def test_prod_docs_404(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_ENABLE_DOCS", raising=False)
        settings = load_settings(env="prod")
        app = create_app(settings)
        jobs_store.stop_worker()
        client = TestClient(app)

        resp = client.get("/docs")
        assert resp.status_code == 404

        resp = client.get("/openapi.json")
        assert resp.status_code == 404

    def test_dev_docs_200(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_ENABLE_DOCS", raising=False)
        settings = load_settings(env="dev")
        app = create_app(settings)
        jobs_store.stop_worker()
        client = TestClient(app)

        resp = client.get("/docs")
        assert resp.status_code == 200


# ── Allow Nonlocal ───────────────────────────────────────────────


class TestAllowNonlocal:
    def test_default_nonlocal_blocked(self):
        with pytest.raises(ValueError, match="non-local"):
            validate_host("0.0.0.0", allow_nonlocal=False)

    def test_env_override_allows_nonlocal(self, monkeypatch):
        monkeypatch.setenv("VERIFILY_ALLOW_NONLOCAL", "1")
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        settings = load_settings()
        assert settings.allow_nonlocal is True
        # Should not raise
        validate_host("0.0.0.0", allow_nonlocal=settings.allow_nonlocal)

    def test_localhost_always_allowed(self):
        # Should not raise for any localhost variant
        validate_host("127.0.0.1", allow_nonlocal=False)
        validate_host("localhost", allow_nonlocal=False)


# ── /ready Stability ────────────────────────────────────────────


class TestReadyStable:
    def test_ready_returns_status(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        app = create_app()
        jobs_store.stop_worker()
        client = TestClient(app)

        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert "checks" in data
