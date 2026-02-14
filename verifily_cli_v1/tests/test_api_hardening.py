"""Tests for the Production Hardening Pass.

Part A: UsageStore persistence (opt-in via VERIFILY_USAGE_PERSIST=1)
Part B: Rate limiting middleware (opt-in via VERIFILY_RATE_LIMIT_RPM)
Part C: Structured JSON logging (opt-in via VERIFILY_LOG_FORMAT=json)
Part D: Normalized error envelope (always-on)

All tests are deterministic and fast (<1s each). No sleeps > 0.05s.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.usage_store import UsageStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"


def _ingest_drill(tmp_path: Path) -> Path:
    from verifily_cli_v1.commands.ingest import ingest

    ingest(
        DRILL_DIR / "raw" / "support_tickets.csv",
        tmp_path / "artifact",
        schema="sft",
        mapping={"question": "subject", "answer": "resolution", "context": "body"},
        extra_tags={"source": "customer_drill"},
    )
    return tmp_path / "artifact" / "dataset.jsonl"


# ── Part A: UsageStore persistence ──────────────────────────────

class TestUsagePersistenceDisabled:
    def test_no_file_when_disabled(self, monkeypatch, tmp_path):
        """With persistence OFF (default), no file is created."""
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_LOG_PATH", raising=False)
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_RATE_LIMIT_RPM", raising=False)

        log_path = tmp_path / "usage.jsonl"
        monkeypatch.setenv("VERIFILY_USAGE_LOG_PATH", str(log_path))
        # Note: NOT setting VERIFILY_USAGE_PERSIST=1

        app = create_app()
        client = TestClient(app)
        client.get("/health")

        assert not log_path.exists()


class TestUsagePersistenceEnabled:
    def test_file_created_on_record(self, monkeypatch, tmp_path):
        """With persistence ON, events are written to the JSONL file."""
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_RATE_LIMIT_RPM", raising=False)

        log_path = tmp_path / "usage.jsonl"
        monkeypatch.setenv("VERIFILY_USAGE_PERSIST", "1")
        monkeypatch.setenv("VERIFILY_USAGE_LOG_PATH", str(log_path))

        train = _ingest_drill(tmp_path)
        app = create_app()
        client = TestClient(app)

        # Make a request that triggers usage recording
        client.post("/v1/report", json={"dataset_path": str(train)})

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        evt = json.loads(lines[0])
        assert "ts" in evt
        assert "api_key_id" in evt
        assert "project_id" in evt
        assert "counters" in evt
        # Privacy: no raw API keys, no payloads
        assert "api_key" not in evt
        assert "body" not in evt

    def test_replay_on_restart(self, monkeypatch, tmp_path):
        """A new app instance replays the JSONL file and shows prior counts."""
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_RATE_LIMIT_RPM", raising=False)

        log_path = tmp_path / "usage.jsonl"
        monkeypatch.setenv("VERIFILY_USAGE_PERSIST", "1")
        monkeypatch.setenv("VERIFILY_USAGE_LOG_PATH", str(log_path))

        train = _ingest_drill(tmp_path)

        # App instance 1: make requests
        app1 = create_app()
        c1 = TestClient(app1)
        c1.post("/v1/report", json={"dataset_path": str(train)})
        c1.post("/v1/report", json={"dataset_path": str(train)})

        # Verify app1 sees 2 requests
        resp1 = c1.get("/v1/usage", params={"group_by": "total"})
        assert resp1.json()["total"]["requests"] == 2

        # App instance 2: should replay from file
        app2 = create_app()
        c2 = TestClient(app2)

        resp2 = c2.get("/v1/usage", params={"group_by": "total"})
        assert resp2.json()["total"]["requests"] == 2


# ── Part B: Rate limiting ───────────────────────────────────────

class TestRateLimitOff:
    def test_limiter_off_requests_pass(self, monkeypatch):
        """When VERIFILY_RATE_LIMIT_RPM is unset, no rate limiting."""
        monkeypatch.delenv("VERIFILY_RATE_LIMIT_RPM", raising=False)
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()
        client = TestClient(app)

        # Many requests should all succeed
        for _ in range(20):
            resp = client.get("/v1/usage")
            assert resp.status_code == 200


class TestRateLimitOn:
    def test_429_after_limit_exceeded(self, monkeypatch):
        """When RPM=3, the 4th request within the window gets 429."""
        monkeypatch.setenv("VERIFILY_RATE_LIMIT_RPM", "3")
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()
        client = TestClient(app)

        results = []
        for _ in range(5):
            resp = client.get("/v1/usage")
            results.append(resp.status_code)

        assert results[:3] == [200, 200, 200]
        assert 429 in results[3:]

    def test_429_has_error_envelope(self, monkeypatch):
        """429 response follows the normalized error envelope."""
        monkeypatch.setenv("VERIFILY_RATE_LIMIT_RPM", "1")
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()
        client = TestClient(app)

        client.get("/v1/usage")  # uses up the limit
        resp = client.get("/v1/usage")  # should be 429

        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["type"] == "RATE_LIMITED"
        assert "message" in body["error"]
        assert "request_id" in body["error"]

    def test_429_has_retry_after_header(self, monkeypatch):
        """429 response includes Retry-After header."""
        monkeypatch.setenv("VERIFILY_RATE_LIMIT_RPM", "1")
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()
        client = TestClient(app)

        client.get("/v1/usage")
        resp = client.get("/v1/usage")

        assert resp.status_code == 429
        retry_after = resp.headers.get("retry-after")
        assert retry_after is not None
        assert int(retry_after) >= 1

    def test_exempt_endpoints_never_limited(self, monkeypatch):
        """/health, /ready, /metrics are never rate-limited."""
        monkeypatch.setenv("VERIFILY_RATE_LIMIT_RPM", "1")
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()
        client = TestClient(app)

        # Use up the limit on /v1/usage
        client.get("/v1/usage")

        # Exempt endpoints should still work
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/metrics").status_code == 200

        # But /v1/* is limited
        resp = client.get("/v1/usage")
        assert resp.status_code == 429


# ── Part C: Structured JSON logging ─────────────────────────────

class TestStructuredLogging:
    def test_json_log_format(self, monkeypatch, caplog):
        """With VERIFILY_LOG_FORMAT=json, log lines parse as JSON with required keys."""
        monkeypatch.setenv("VERIFILY_LOG_FORMAT", "json")
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_RATE_LIMIT_RPM", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="verifily.api"):
            client.get("/health", headers={"X-Request-ID": "json-test-1"})

        # Find the JSON log line for our request
        json_logs = []
        for record in caplog.records:
            if record.name == "verifily.api":
                try:
                    parsed = json.loads(record.message)
                    json_logs.append(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass

        assert len(json_logs) >= 1
        log = json_logs[-1]
        assert log["request_id"] == "json-test-1"
        required_keys = {"ts", "level", "request_id", "api_key_id", "method", "path", "status", "elapsed_ms"}
        assert required_keys.issubset(log.keys())
        assert log["method"] == "GET"
        assert log["path"] == "/health"
        assert log["status"] == 200

    def test_text_log_format_default(self, monkeypatch, caplog):
        """Default (text) log format emits non-JSON log lines."""
        monkeypatch.delenv("VERIFILY_LOG_FORMAT", raising=False)
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_RATE_LIMIT_RPM", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="verifily.api"):
            client.get("/health", headers={"X-Request-ID": "text-test-1"})

        log_messages = [r.message for r in caplog.records if r.name == "verifily.api"]
        assert any("request_id=text-test-1" in m for m in log_messages)
        # Ensure it's NOT JSON
        for msg in log_messages:
            if "text-test-1" in msg:
                assert not msg.startswith("{")


# ── Part D: Normalized error envelope ───────────────────────────

class TestErrorEnvelope:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_RATE_LIMIT_RPM", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)
        return TestClient(create_app())

    def test_401_envelope_with_request_id(self, monkeypatch):
        """401 returns the normalized error envelope with request_id."""
        monkeypatch.setenv("VERIFILY_API_KEY", "correct-key")
        monkeypatch.delenv("VERIFILY_RATE_LIMIT_RPM", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()
        client = TestClient(app)

        resp = client.get(
            "/v1/usage",
            headers={"X-Request-ID": "auth-test-42"},
        )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["type"] == "AUTH_ERROR"
        assert "message" in body["error"]
        assert body["error"]["request_id"] == "auth-test-42"
        # request_id also in response header
        assert resp.headers["x-request-id"] == "auth-test-42"

    def test_422_envelope_with_request_id(self, client):
        """422 returns the normalized error envelope with request_id."""
        resp = client.post(
            "/v1/pipeline",
            json={"plan": True},
            headers={"X-Request-ID": "val-test-99"},
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["type"] == "VALIDATION_ERROR"
        assert "message" in body["error"]
        assert body["error"]["request_id"] == "val-test-99"

    def test_404_envelope_with_request_id(self, client):
        """404 returns the normalized error envelope with request_id."""
        resp = client.post(
            "/v1/pipeline",
            json={"config_path": "/nonexistent/verifily.yaml", "plan": True},
            headers={"X-Request-ID": "not-found-77"},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["type"] == "NOT_FOUND"
        assert "message" in body["error"]
        assert body["error"]["request_id"] == "not-found-77"

    def test_500_envelope_with_request_id(self, monkeypatch):
        """500 returns the normalized error envelope with request_id."""
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_RATE_LIMIT_RPM", raising=False)
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()

        # Add a test-only route that always raises
        @app.get("/v1/test-500")
        def explode():
            raise RuntimeError("boom")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/test-500",
            headers={"X-Request-ID": "err-test-500"},
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["type"] == "INTERNAL_ERROR"
        assert "message" in body["error"]
        assert body["error"]["request_id"] == "err-test-500"

    def test_envelope_consistent_across_error_types(self, monkeypatch):
        """All error types share the same {"error": {"type", "message", "request_id"}} shape."""
        monkeypatch.setenv("VERIFILY_API_KEY", "right-key")
        monkeypatch.setenv("VERIFILY_RATE_LIMIT_RPM", "1")
        monkeypatch.delenv("VERIFILY_USAGE_PERSIST", raising=False)

        app = create_app()
        client = TestClient(app)

        # 401 — no auth header
        resp_401 = client.get("/v1/usage")
        # 429 — use up limit then retry (with auth this time)
        resp_ok = client.get("/v1/usage", headers={"Authorization": "Bearer right-key"})
        resp_429 = client.get("/v1/usage", headers={"Authorization": "Bearer right-key"})

        for resp in [resp_401, resp_429]:
            body = resp.json()
            err = body.get("error")
            assert isinstance(err, dict), f"status {resp.status_code}: 'error' should be dict"
            assert "type" in err
            assert "message" in err
            assert "request_id" in err
