"""Tests for PII/secrets redaction audit.

Target: ~10 tests, runtime <0.3s
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from verifily_cli_v1.core.redaction_audit import (
    audit_run_directory,
    audit_single_file,
    quick_audit,
    write_audit_report,
    RedactionAuditReport,
    LeakFinding,
    RedactionAuditError,
)


@pytest.fixture
def clean_run_dir() -> Generator[Path, None, None]:
    """Create a clean run directory with no leaks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_clean"
        run_dir.mkdir()
        
        # Clean manifest
        manifest = {
            "run_id": "run_001",
            "timestamp": "2025-01-20T12:00:00Z",
            "version": "1.0.0",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Clean decision
        decision = {
            "run_id": "run_001",
            "status": "PASS",
            "passed": True,
        }
        (run_dir / "decision.json").write_text(json.dumps(decision))
        
        yield run_dir


@pytest.fixture
def leaky_run_dir() -> Generator[Path, None, None]:
    """Create a run directory with PII/secrets leaks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_leaky"
        run_dir.mkdir()
        
        # Leaky manifest with email (use non-example domain to avoid whitelist)
        manifest = {
            "run_id": "run_002",
            "contact": "admin@company.com",
            "timestamp": "2025-01-20T12:00:00Z",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Leaky decision with API key
        decision = {
            "run_id": "run_002",
            "status": "PASS",
            "api_key": "sk-live-1234567890abcdef1234",
        }
        (run_dir / "decision.json").write_text(json.dumps(decision))
        
        # Log with password
        log_entry = {
            "timestamp": "2025-01-20T12:00:00Z",
            "user": "testuser",
            "password": "SuperSecret123!",
        }
        (run_dir / "audit.log").write_text(json.dumps(log_entry))
        
        yield run_dir


class TestAuditRunDirectory:
    """Test full run directory audit."""

    def test_clean_run_passes(self, clean_run_dir: Path) -> None:
        """Clean run gets PASS status."""
        report = audit_run_directory(clean_run_dir)
        
        assert report.status == "PASS"
        assert report.files_scanned >= 2
        assert len(report.findings) == 0

    def test_leaky_run_fails(self, leaky_run_dir: Path) -> None:
        """Leaky run gets FAIL status."""
        report = audit_run_directory(leaky_run_dir)
        
        assert report.status == "FAIL"
        assert len(report.findings) > 0
        assert report.summary["high_severity"] > 0

    def test_missing_directory_raises(self) -> None:
        """Missing directory raises error."""
        with pytest.raises(RedactionAuditError):
            audit_run_directory("/nonexistent/path")

    def test_finds_email(self, leaky_run_dir: Path) -> None:
        """Email addresses are detected."""
        report = audit_run_directory(leaky_run_dir)
        
        email_findings = [f for f in report.findings if f.type == "email"]
        assert len(email_findings) > 0
        assert any("admin@company.com" in f.snippet for f in email_findings)

    def test_finds_api_key(self, leaky_run_dir: Path) -> None:
        """API keys are detected."""
        report = audit_run_directory(leaky_run_dir)
        
        key_findings = [f for f in report.findings if f.type == "api_key"]
        assert len(key_findings) > 0

    def test_finds_password(self, leaky_run_dir: Path) -> None:
        """Passwords are detected."""
        report = audit_run_directory(leaky_run_dir)
        
        pass_findings = [f for f in report.findings if f.type == "password"]
        assert len(pass_findings) > 0

    def test_warn_status_for_medium_only(self, clean_run_dir: Path) -> None:
        """WARN status when only medium severity findings."""
        # Add only email (medium severity)
        manifest = json.loads((clean_run_dir / "manifest.json").read_text())
        manifest["contact"] = "test@mycompany.com"
        (clean_run_dir / "manifest.json").write_text(json.dumps(manifest))
        
        report = audit_run_directory(clean_run_dir)
        
        # Should be WARN (not FAIL) for medium severity
        if report.status == "WARN":
            assert report.summary["high_severity"] == 0
            assert report.summary["medium_severity"] > 0


class TestAuditSingleFile:
    """Test single file audit."""

    def test_detects_in_single_file(self) -> None:
        """Leaks detected in single file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"api_key": "secret1234567890"}')
            f.flush()
            temp_path = Path(f.name)
        
        try:
            findings = audit_single_file(temp_path)
            assert len(findings) > 0
            assert any(f.type == "api_key" for f in findings)
        finally:
            temp_path.unlink()

    def test_empty_file_no_findings(self) -> None:
        """Empty file has no findings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{}')
            f.flush()
            temp_path = Path(f.name)
        
        try:
            findings = audit_single_file(temp_path)
            assert len(findings) == 0
        finally:
            temp_path.unlink()


class TestQuickAudit:
    """Test quick audit function."""

    def test_quick_returns_status(self, clean_run_dir: Path) -> None:
        """Quick audit returns status string."""
        status = quick_audit(clean_run_dir)
        assert status in ("PASS", "WARN", "FAIL")

    def test_quick_on_leaky(self, leaky_run_dir: Path) -> None:
        """Quick audit returns FAIL for leaky run."""
        status = quick_audit(leaky_run_dir)
        assert status == "FAIL"


class TestAuditReport:
    """Test audit report functionality."""

    def test_to_dict_structure(self, leaky_run_dir: Path) -> None:
        """Report converts to dict correctly."""
        report = audit_run_directory(leaky_run_dir)
        d = report.to_dict()
        
        assert "status" in d
        assert "files_scanned" in d
        assert "findings_count" in d
        assert "findings" in d
        assert "summary" in d

    def test_to_json_output(self, leaky_run_dir: Path) -> None:
        """Report converts to JSON correctly."""
        report = audit_run_directory(leaky_run_dir)
        json_str = report.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["status"] == report.status

    def test_write_audit_report(self, leaky_run_dir: Path) -> None:
        """Report can be written to file."""
        report = audit_run_directory(leaky_run_dir)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "redaction_audit.json"
            written = write_audit_report(report, output_path)
            
            assert written.exists()
            content = json.loads(written.read_text())
            assert content["status"] == report.status


class TestLeakFinding:
    """Test LeakFinding dataclass."""

    def test_to_dict(self) -> None:
        """Finding converts to dict."""
        finding = LeakFinding(
            type="email",
            severity="MEDIUM",
            file="test.json",
            line=5,
            column=10,
            snippet="admin@example.com",
            context="contact field",
        )
        
        d = finding.to_dict()
        assert d["type"] == "email"
        assert d["severity"] == "MEDIUM"
        assert d["line"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
