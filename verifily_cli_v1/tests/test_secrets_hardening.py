"""Tests for secrets redaction, config overrides, privacy modes, and security checks."""

from __future__ import annotations

import json
import os
import copy
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from verifily_cli_v1.cli import app
from verifily_cli_v1.core.secrets import (
    REDACTED,
    SENSITIVE_KEYWORDS,
    assert_no_secrets,
    check_privacy_mode,
    load_dotenv_if_present,
    redact_dict,
    redact_text,
    safe_log_json,
    scan_directory_for_secrets,
)
from verifily_cli_v1.core.config_overrides import resolve_config
from verifily_cli_v1.core.env import (
    check_auth_configured,
    check_log_format,
    check_privacy_mode as check_privacy_mode_env,
)

runner = CliRunner()


# ── redact_dict ──────────────────────────────────────────────────


class TestRedactDict:
    def test_redacts_api_key(self):
        data = {"api_key": "sk-secret-123"}
        result = redact_dict(data)
        assert result["api_key"] == REDACTED

    def test_redacts_nested_token(self):
        data = {"auth": {"bearer_token": "abc123"}}
        result = redact_dict(data)
        assert result["auth"]["bearer_token"] == REDACTED

    def test_redacts_case_insensitive(self):
        data = {"API_KEY": "secret", "Authorization": "value"}
        result = redact_dict(data)
        assert result["API_KEY"] == REDACTED
        assert result["Authorization"] == REDACTED

    def test_does_not_mutate_input(self):
        data = {"api_key": "original"}
        original = copy.deepcopy(data)
        redact_dict(data)
        assert data == original

    def test_truncates_long_strings(self):
        data = {"description": "x" * 300}
        result = redact_dict(data)
        assert len(result["description"]) < 300
        assert "..." in result["description"]

    def test_handles_non_dict_input(self):
        assert redact_dict("hello") == "hello"
        assert redact_dict(42) == 42
        assert redact_dict(None) is None
        assert redact_dict([1, 2, 3]) == [1, 2, 3]


# ── redact_text ──────────────────────────────────────────────────


class TestRedactText:
    def test_redacts_bearer_token(self):
        text = "Authorization: Bearer sk-abc123def456"
        result = redact_text(text)
        assert "sk-abc123def456" not in result
        assert REDACTED in result

    def test_redacts_long_hex(self):
        text = "key=aabbccddee112233445566778899aabb"
        result = redact_text(text)
        assert "aabbccddee112233445566778899aabb" not in result

    def test_preserves_normal_text(self):
        text = "This is a normal log message"
        assert redact_text(text) == text

    def test_empty_string(self):
        assert redact_text("") == ""


# ── safe_log_json ────────────────────────────────────────────────


class TestSafeLogJson:
    def test_removes_dataset_keys(self):
        event = {"status": "ok", "rows": [1, 2, 3], "dataset": {"big": "data"}}
        result = safe_log_json(event)
        assert "rows" not in result
        assert "dataset" not in result
        assert result["status"] == "ok"

    def test_enforces_size_limit(self):
        event = {"big_value": "x" * 20000}
        result = safe_log_json(event)
        serialized = json.dumps(result, separators=(",", ":"), default=str)
        assert len(serialized) <= 8192

    def test_redacts_sensitive_keys(self):
        event = {"api_key": "sk-secret", "status": "ok"}
        result = safe_log_json(event)
        assert result["api_key"] == REDACTED


# ── assert_no_secrets ────────────────────────────────────────────


class TestAssertNoSecrets:
    def test_clean_dict_passes(self):
        data = {"status": "ok", "count": 5}
        assert_no_secrets(data)  # Should not raise

    def test_catches_bearer_token(self):
        data = {"message": "Auth: Bearer sk-abc123def456"}
        with pytest.raises(ValueError, match="secret_leak_detected"):
            assert_no_secrets(data)

    def test_catches_api_key_value(self):
        data = {"api_key": "sk-secret-value-here"}
        with pytest.raises(ValueError, match="secret_leak_detected"):
            assert_no_secrets(data)


# ── config overrides ─────────────────────────────────────────────


class TestConfigOverrides:
    def test_env_overrides_applied(self):
        base = {"privacy_mode": "local"}
        env = {"VERIFILY_PRIVACY_MODE": "hybrid"}
        result = resolve_config(base, env)
        assert result["privacy_mode"] == "hybrid"

    def test_request_overrides_applied(self):
        base = {"privacy_mode": "local"}
        env = {}
        result = resolve_config(base, env, {"privacy_mode": "remote"})
        assert result["privacy_mode"] == "remote"

    def test_api_key_never_in_output(self):
        base = {}
        env = {"VERIFILY_API_KEY": "sk-secret-123"}
        result = resolve_config(base, env)
        assert "VERIFILY_API_KEY" not in result
        assert "verifily_api_key" not in result
        for v in result.values():
            if isinstance(v, str):
                assert "sk-secret-123" not in v

    def test_unknown_env_ignored(self):
        base = {"mode": "test"}
        env = {"UNKNOWN_VAR": "value"}
        result = resolve_config(base, env)
        assert "UNKNOWN_VAR" not in result
        assert result["mode"] == "test"


# ── privacy mode ─────────────────────────────────────────────────


class TestPrivacyMode:
    def test_local_forbids_remote_keys(self):
        config = {"remote_url": "https://example.com"}
        with pytest.raises(ValueError, match="LOCAL mode forbids"):
            check_privacy_mode(config, "local")

    def test_local_allows_clean_config(self):
        config = {"privacy_mode": "local", "rate_limit_rpm": 100}
        check_privacy_mode(config, "local")  # Should not raise


# ── effective config API ─────────────────────────────────────────


class TestEffectiveConfigAPI:
    @pytest.fixture
    def api_client(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        from verifily_cli_v1.core.api.server import create_app
        from verifily_cli_v1.core.api.jobs import jobs_store
        a = create_app()
        jobs_store.stop_worker()
        return TestClient(a)

    def test_returns_redacted_config(self, api_client):
        resp = api_client.get("/v1/config/effective")
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "privacy_mode" in data["config"]

    def test_no_api_key_in_response(self, api_client, monkeypatch):
        monkeypatch.setenv("VERIFILY_API_KEY", "sk-secret-key")
        resp = api_client.get("/v1/config/effective")
        data = resp.json()
        config_str = json.dumps(data)
        assert "sk-secret-key" not in config_str


# ── security-check CLI ───────────────────────────────────────────


class TestSecurityCheck:
    def test_clean_dir_exit_0(self, tmp_path):
        clean_file = tmp_path / "clean.json"
        clean_file.write_text('{"status": "ok"}')
        result = runner.invoke(app, ["security-check", str(tmp_path)])
        assert result.exit_code == 0

    def test_leaked_secret_exit_1(self, tmp_path):
        leak_file = tmp_path / "config.json"
        leak_file.write_text('{"api_key": "Bearer sk-aabbccddee112233445566778899aabb00"}')
        result = runner.invoke(app, ["security-check", str(tmp_path)])
        assert result.exit_code == 1


# ── doctor security checks ──────────────────────────────────────


class TestDoctorSecurityChecks:
    def test_log_format_check(self, monkeypatch):
        monkeypatch.setenv("VERIFILY_LOG_FORMAT", "json")
        check = check_log_format()
        assert check.status == "PASS"

        monkeypatch.setenv("VERIFILY_LOG_FORMAT", "text")
        check = check_log_format()
        assert check.status == "WARN"

    def test_auth_check(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        check = check_auth_configured()
        assert check.status == "WARN"

        monkeypatch.setenv("VERIFILY_API_KEY", "sk-test")
        check = check_auth_configured()
        assert check.status == "PASS"
