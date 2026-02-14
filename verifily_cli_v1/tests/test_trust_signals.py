"""Tests for trust, security, and reliability signals."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from verifily_cli_v1 import __version__
from verifily_cli_v1.cli import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestSecurityDocs:
    def test_security_md_exists(self):
        assert (REPO_ROOT / "SECURITY.md").is_file()

    def test_security_md_has_reporting_section(self):
        text = (REPO_ROOT / "SECURITY.md").read_text()
        assert "Reporting" in text

    def test_security_md_has_contact(self):
        text = (REPO_ROOT / "SECURITY.md").read_text()
        assert "security@verifily.dev" in text


class TestGuaranteesDocs:
    def test_guarantees_md_exists(self):
        assert (REPO_ROOT / "docs" / "guarantees.md").is_file()

    def test_guarantees_md_has_exit_codes(self):
        text = (REPO_ROOT / "docs" / "guarantees.md").read_text()
        assert "SHIP" in text
        assert "DONT_SHIP" in text

    def test_guarantees_md_has_determinism(self):
        text = (REPO_ROOT / "docs" / "guarantees.md").read_text()
        assert "Determinism" in text


class TestThreatModelDocs:
    def test_threat_model_md_exists(self):
        assert (REPO_ROOT / "docs" / "threat_model.md").is_file()

    def test_threat_model_has_trust_boundaries(self):
        text = (REPO_ROOT / "docs" / "threat_model.md").read_text()
        assert "Trust Boundaries" in text


class TestReliabilityDocs:
    def test_reliability_md_exists(self):
        assert (REPO_ROOT / "docs" / "reliability.md").is_file()

    def test_reliability_has_exit_codes_contract(self):
        text = (REPO_ROOT / "docs" / "reliability.md").read_text()
        assert "Exit Codes Contract" in text

    def test_reliability_has_determinism(self):
        text = (REPO_ROOT / "docs" / "reliability.md").read_text()
        assert "Determinism" in text

    def test_reliability_has_artifact_guarantees(self):
        text = (REPO_ROOT / "docs" / "reliability.md").read_text()
        assert "Artifact Write Guarantees" in text


class TestCLITrustMessaging:
    def test_help_footer_contains_version(self):
        result = runner.invoke(app, ["--help"])
        assert __version__ in result.output

    def test_help_footer_contains_deterministic(self):
        result = runner.invoke(app, ["--help"])
        assert "Deterministic" in result.output

    def test_help_references_security_docs(self):
        result = runner.invoke(app, ["--help"])
        assert "SECURITY.md" in result.output


class TestHealthEndpoint:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from verifily_cli_v1.core.api.server import create_app
        return TestClient(create_app())

    def test_health_contains_version(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == __version__

    def test_health_contains_mode(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["mode"] == "local"

    def test_health_contains_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_contains_time(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "time" in data
        assert data["time"].endswith("Z")


class TestReadyEndpoint:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from verifily_cli_v1.core.api.server import create_app
        return TestClient(create_app())

    def test_ready_contains_components(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        checks = data["checks"]
        assert "python" in checks
        assert "temp_write" in checks
        assert "imports" in checks

    def test_ready_contains_store_status(self, client):
        resp = client.get("/ready")
        data = resp.json()
        checks = data["checks"]
        assert checks["jobs_store"] == "ok"
        assert checks["usage_store"] == "ok"
        assert checks["monitor_store"] == "ok"

    def test_ready_status_is_ready(self, client):
        resp = client.get("/ready")
        data = resp.json()
        assert data["status"] == "ready"
