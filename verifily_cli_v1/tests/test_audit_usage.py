"""Tests for audit log and usage metering.

Covers:
  - AuditLogger: event accumulation, event ordering, write to JSONL
  - UsageMeter: counter accumulation, timing, to_dict determinism
  - Pipeline integration: audit_log.jsonl + usage.json written when output_dir set
  - Plan mode: usage computed in memory, no files written
  - API mode: request_id flows through to audit/usage
  - PII safety: no raw data in audit events
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifily_cli_v1.core.audit import AuditEvent, AuditLogger
from verifily_cli_v1.core.usage import UsageMeter

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


def _write_pipeline_config(tmp_path: Path, train_path: Path, eval_file: str, run_name: str) -> Path:
    from verifily_cli_v1.core.io import write_yaml

    cfg = {
        "run_dir": str(DRILL_DIR / "runs" / run_name),
        "train_data": str(train_path),
        "eval_data": str(DRILL_DIR / "raw" / eval_file),
        "baseline_run": str(DRILL_DIR / "runs" / run_name),
        "ship_if": {
            "min_f1": 0.65, "min_exact_match": 0.50,
            "max_f1_regression": 0.03, "max_pii_hits": 10,
        },
    }
    config_path = tmp_path / "pipeline.yaml"
    write_yaml(config_path, cfg)
    return config_path


# ── AuditLogger unit tests ──────────────────────────────────────

class TestAuditLogger:
    def test_events_accumulate_in_order(self):
        audit = AuditLogger(run_id="test-run-1")
        audit.start("CONTRACT")
        audit.ok("CONTRACT", elapsed_ms=5)
        audit.start("REPORT")
        audit.ok("REPORT", elapsed_ms=10)

        events = audit.events
        assert len(events) == 4
        assert events[0]["step"] == "CONTRACT"
        assert events[0]["status"] == "START"
        assert events[1]["step"] == "CONTRACT"
        assert events[1]["status"] == "OK"
        assert events[2]["step"] == "REPORT"
        assert events[3]["step"] == "REPORT"

    def test_event_fields_present(self):
        audit = AuditLogger(run_id="r1", request_id="req-42", project="proj")
        audit.start("CONTRACT", inputs={"run_dir": "/tmp/run"})
        audit.ok("CONTRACT", elapsed_ms=3, summary={"valid": True})

        events = audit.events
        start = events[0]
        assert start["run_id"] == "r1"
        assert start["request_id"] == "req-42"
        assert start["project"] == "proj"
        assert start["inputs"] == {"run_dir": "/tmp/run"}
        assert "ts" in start

        ok = events[1]
        assert ok["elapsed_ms"] == 3
        assert ok["summary"] == {"valid": True}

    def test_fail_and_warn_statuses(self):
        audit = AuditLogger(run_id="r2")
        audit.fail("CONTRACT", exit_code=3, elapsed_ms=2, summary={"valid": False})
        audit.warn("REPORT", elapsed_ms=5, summary={"pii_clean": False})

        events = audit.events
        assert events[0]["status"] == "FAIL"
        assert events[0]["exit_code"] == 3
        assert events[1]["status"] == "WARN"

    def test_write_jsonl(self, tmp_path):
        audit = AuditLogger(run_id="r3")
        audit.start("CONTRACT")
        audit.ok("CONTRACT", elapsed_ms=1)

        path = audit.write(tmp_path)
        assert path == tmp_path / "audit_log.jsonl"
        assert path.exists()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "step" in obj
            assert "status" in obj
            assert "run_id" in obj

    def test_no_raw_data_in_events(self):
        """Audit events should never contain raw dataset rows or PII values."""
        audit = AuditLogger(run_id="r4")
        audit.ok("REPORT", elapsed_ms=5, summary={"row_count": 100, "pii_clean": True})

        events = audit.events
        for event in events:
            # summary should only contain aggregate counts, not raw text
            summary = event.get("summary", {})
            for key in summary:
                assert key in ("row_count", "pii_clean", "pii_total_hits", "valid",
                               "status", "exact_overlaps", "near_duplicates",
                               "recommendation", "exit_code", "confidence"), \
                    f"Unexpected summary key: {key}"


# ── UsageMeter unit tests ────────────────────────────────────────

class TestUsageMeter:
    def test_basic_accumulation(self):
        meter = UsageMeter(run_id="u1", mode="cli", ci=False)
        meter.record_contract(valid=True, elapsed_ms=5)
        meter.record_report(rows_in=100, bytes_in=4096, elapsed_ms=12)
        meter.record_contamination(status="PASS", checked_rows=200, elapsed_ms=8)
        meter.record_decision(decision="SHIP", exit_code=0, elapsed_ms=2)
        meter.finalize(total_elapsed_ms=30)

        d = meter.to_dict()
        assert d["run_id"] == "u1"
        assert d["mode"] == "cli"
        assert d["ci"] is False

        units = d["billable_units"]
        assert units["rows_in"] == 100
        assert units["bytes_in"] == 4096
        assert units["rows_out"] == 200
        assert units["contamination_checks"] == 1
        assert units["reports_generated"] == 1
        assert units["decisions_generated"] == 1
        assert units["contracts_validated"] == 1

        timing = d["timing_ms"]
        assert timing["contract"] == 5
        assert timing["report"] == 12
        assert timing["contamination"] == 8
        assert timing["decision"] == 2
        assert timing["total"] == 30

        result = d["result"]
        assert result["decision"] == "SHIP"
        assert result["exit_code"] == 0
        assert result["contamination_status"] == "PASS"
        assert result["contract_valid"] is True

    def test_request_id_and_privacy(self):
        meter = UsageMeter(run_id="u2", request_id="req-99", privacy="redacted")
        d = meter.to_dict()
        assert d["request_id"] == "req-99"
        assert d["privacy"] == "redacted"

    def test_deterministic_except_timestamps(self):
        """Same inputs produce same usage dict (no timestamps in usage)."""
        def _build():
            m = UsageMeter(run_id="det", mode="cli", ci=True)
            m.record_contract(valid=True, elapsed_ms=5)
            m.record_report(rows_in=50, bytes_in=2048, elapsed_ms=10)
            m.record_contamination(status="PASS", checked_rows=100, elapsed_ms=7)
            m.record_decision(decision="SHIP", exit_code=0, elapsed_ms=1)
            m.finalize(total_elapsed_ms=25)
            return m.to_dict()

        d1 = _build()
        d2 = _build()
        assert d1 == d2


# ── Pipeline integration tests ──────────────────────────────────

class TestPipelineAuditUsage:
    def test_pipeline_writes_audit_and_usage(self, tmp_path):
        """Normal pipeline run writes audit_log.jsonl and usage.json."""
        from verifily_cli_v1.commands.pipeline import run_pipeline

        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        output = tmp_path / "output"

        result = run_pipeline(config, ci=True, output_dir=str(output))

        # Artifacts written
        assert (output / "audit_log.jsonl").exists()
        assert (output / "usage.json").exists()
        assert (output / "pipeline_result.json").exists()

        # Audit log has at least START+OK for each of the 4 steps
        lines = (output / "audit_log.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        steps = [e["step"] for e in events]
        assert "CONTRACT" in steps
        assert "REPORT" in steps
        assert "CONTAMINATION" in steps
        assert "DECISION" in steps

        # All events have the same run_id
        run_ids = {e["run_id"] for e in events}
        assert len(run_ids) == 1

        # Usage JSON valid
        usage = json.loads((output / "usage.json").read_text())
        assert usage["run_id"] == events[0]["run_id"]
        assert usage["mode"] == "cli"
        assert usage["billable_units"]["rows_in"] > 0
        assert usage["billable_units"]["contracts_validated"] == 1
        assert usage["timing_ms"]["total"] > 0

    def test_pipeline_result_includes_usage(self, tmp_path):
        """Pipeline result dict includes usage data."""
        from verifily_cli_v1.commands.pipeline import run_pipeline

        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")

        result = run_pipeline(config, ci=True)

        assert "usage" in result
        assert "run_id" in result["usage"]
        assert result["usage"]["result"]["decision"] == "SHIP"

    def test_pipeline_no_output_dir_no_files(self, tmp_path):
        """When output_dir is None, no audit/usage files are written."""
        from verifily_cli_v1.commands.pipeline import run_pipeline

        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")

        result = run_pipeline(config, ci=True, output_dir=None)

        # Usage still in result dict
        assert "usage" in result
        # But no files written to tmp_path (except the config and ingest artifacts)
        assert not (tmp_path / "audit_log.jsonl").exists()
        assert not (tmp_path / "usage.json").exists()

    def test_pipeline_early_exit_contract_fail_still_records_usage(self, tmp_path):
        """When contract fails (early exit), usage is still recorded."""
        from verifily_cli_v1.commands.pipeline import run_pipeline
        from verifily_cli_v1.core.io import write_yaml

        train = _ingest_drill(tmp_path)
        config = tmp_path / "bad_pipeline.yaml"
        write_yaml(config, {
            "run_dir": str(tmp_path / "nonexistent_run"),
            "train_data": str(train),
            "eval_data": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
            "ship_if": {"min_f1": 0.65},
        })
        output = tmp_path / "output_fail"

        result = run_pipeline(config, ci=True, output_dir=str(output))

        assert result["decision"]["recommendation"] == "DONT_SHIP"
        assert "usage" in result
        assert result["usage"]["result"]["decision"] == "DONT_SHIP"
        assert result["usage"]["billable_units"]["contracts_validated"] == 1
        # Audit written
        assert (output / "audit_log.jsonl").exists()
        assert (output / "usage.json").exists()

    def test_pipeline_request_id_flows_through(self, tmp_path):
        """request_id and mode params flow into audit and usage."""
        from verifily_cli_v1.commands.pipeline import run_pipeline

        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        output = tmp_path / "output_reqid"

        result = run_pipeline(
            config, ci=True, output_dir=str(output),
            request_id="test-req-123", mode="api",
        )

        # Check usage
        assert result["usage"]["request_id"] == "test-req-123"
        assert result["usage"]["mode"] == "api"

        # Check audit events
        lines = (output / "audit_log.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        for event in events:
            assert event["request_id"] == "test-req-123"

    def test_leaked_pipeline_writes_audit(self, tmp_path):
        """Contamination-failed pipeline still produces complete audit trail."""
        from verifily_cli_v1.commands.pipeline import run_pipeline

        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_leaked_exact.jsonl", "run_leaked")
        output = tmp_path / "output_leaked"

        result = run_pipeline(config, ci=True, output_dir=str(output))

        assert result["decision"]["recommendation"] == "DONT_SHIP"

        # Audit has all 4 steps
        lines = (output / "audit_log.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        steps = {e["step"] for e in events}
        assert steps == {"CONTRACT", "REPORT", "CONTAMINATION", "DECISION"}

        # Usage records contamination failure
        usage = json.loads((output / "usage.json").read_text())
        assert usage["result"]["contamination_status"] == "FAIL"
        assert usage["result"]["decision"] == "DONT_SHIP"


# ── API integration tests ────────────────────────────────────────

class TestAPIAuditUsage:
    def test_api_pipeline_includes_usage(self, tmp_path, monkeypatch):
        """API pipeline response includes usage with request_id."""
        from fastapi.testclient import TestClient
        from verifily_cli_v1.core.api.server import create_app

        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        client = TestClient(create_app())

        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")

        resp = client.post(
            "/v1/pipeline",
            json={"config_path": str(config), "plan": False, "ci": True},
            headers={"X-Request-ID": "api-test-req-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "usage" in data
        assert data["usage"]["mode"] == "api"
        assert data["usage"]["request_id"] == "api-test-req-1"

    def test_api_plan_mode_still_has_usage(self, tmp_path, monkeypatch):
        """Plan mode: usage in response, but no output_dir → no files."""
        from fastapi.testclient import TestClient
        from verifily_cli_v1.core.api.server import create_app

        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        client = TestClient(create_app())

        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")

        resp = client.post(
            "/v1/pipeline",
            json={"config_path": str(config), "plan": True, "ci": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "usage" in data
        assert data["output_dir"] is None
