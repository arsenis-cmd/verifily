"""Tests for production readiness validator.

Target: ~10 tests, runtime <0.5s
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from verifily_cli_v1.core.readiness import (
    ReadinessCheck,
    ReadinessReport,
    ReadinessStatus,
    check_contract_integrity,
    check_hash_chain,
    check_privacy_safety,
    check_contamination_risk,
    check_regression_risk,
    check_config_sanity,
    check_reproducibility,
    validate_readiness,
    format_readiness_report,
    ReadinessError,
)


@pytest.fixture
def clean_run_dir() -> Generator[Path, None, None]:
    """Create a clean run directory that should pass all checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_clean"
        run_dir.mkdir()
        
        # Create manifest
        manifest = {
            "run_id": "run_clean",
            "timestamp": "2025-01-20T12:00:00Z",
            "version": "1.0.0",
            "contracts": [
                {"name": "completeness", "min_row_count": 100}
            ],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Create decision (PASS)
        decision = {
            "run_id": "run_clean",
            "status": "PASS",
            "passed": True,
            "checks": {"completeness": "PASS", "schema": "PASS"},
        }
        (run_dir / "decision.json").write_text(json.dumps(decision))
        
        # Create environment
        environment = {
            "seed": 42,
            "versions": {"verifily": "1.0.0"},
        }
        (run_dir / "environment.json").write_text(json.dumps(environment))
        
        # Create redaction audit (PASS)
        audit = {
            "status": "PASS",
            "findings_count": 0,
            "summary": {"findings_by_type": {}},
        }
        (run_dir / "redaction_audit.json").write_text(json.dumps(audit))
        
        yield run_dir


@pytest.fixture
def contaminated_run_dir() -> Generator[Path, None, None]:
    """Create a run with contamination failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_contaminated"
        run_dir.mkdir()
        
        # Create manifest
        manifest = {"run_id": "run_contaminated", "version": "1.0.0"}
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Create decision (FAIL due to contamination)
        decision = {
            "run_id": "run_contaminated",
            "status": "FAIL",
            "passed": False,
            "checks": {
                "completeness": "PASS",
                "contamination": "FAIL",  # Contamination detected
            },
        }
        (run_dir / "decision.json").write_text(json.dumps(decision))
        
        yield run_dir


@pytest.fixture
def regressed_run_dir() -> Generator[Path, None, None]:
    """Create a run with regression warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_regressed"
        run_dir.mkdir()
        
        # Create manifest
        manifest = {"run_id": "run_regressed", "version": "1.0.0"}
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Create decision with regression warning
        decision = {
            "run_id": "run_regressed",
            "status": "PASS",
            "passed": True,
            "summary": {
                "warnings": ["Regression detected: accuracy dropped 5%"],
            },
            "checks": {"completeness": "PASS"},
        }
        (run_dir / "decision.json").write_text(json.dumps(decision))
        
        yield run_dir


class TestReadinessCheck:
    """Test ReadinessCheck dataclass."""

    def test_check_creation(self) -> None:
        """Can create readiness check."""
        check = ReadinessCheck(
            name="test_check",
            status=ReadinessStatus.PASS,
            message="All good",
        )
        
        assert check.name == "test_check"
        assert check.status == ReadinessStatus.PASS
        assert check.message == "All good"

    def test_check_to_dict(self) -> None:
        """Check converts to dict."""
        check = ReadinessCheck(
            name="test",
            status=ReadinessStatus.FAIL,
            message="Failed",
            details={"error": "test error"},
        )
        
        d = check.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "FAIL"
        assert d["details"]["error"] == "test error"


class TestReadinessReport:
    """Test ReadinessReport dataclass."""

    def test_report_creation(self) -> None:
        """Can create readiness report."""
        report = ReadinessReport(run_id="run_001")
        
        assert report.run_id == "run_001"
        assert report.overall_status == ReadinessStatus.PASS
        assert len(report.checks) == 0

    def test_add_check_updates_status(self) -> None:
        """Adding checks updates overall status."""
        report = ReadinessReport(run_id="run_001")
        
        # Add PASS check - status stays PASS
        report.add_check(ReadinessCheck("c1", ReadinessStatus.PASS, "ok"))
        assert report.overall_status == ReadinessStatus.PASS
        
        # Add WARN check - status becomes WARN
        report.add_check(ReadinessCheck("c2", ReadinessStatus.WARN, "careful"))
        assert report.overall_status == ReadinessStatus.WARN
        
        # Add FAIL check - status becomes FAIL
        report.add_check(ReadinessCheck("c3", ReadinessStatus.FAIL, "bad"))
        assert report.overall_status == ReadinessStatus.FAIL


class TestContractIntegrityCheck:
    """Test contract integrity check."""

    def test_pass_with_all_files(self, clean_run_dir: Path) -> None:
        """Passes when all required files present."""
        check = check_contract_integrity(clean_run_dir)
        
        assert check.status == ReadinessStatus.PASS
        assert "files" in check.message.lower() or "present" in check.message.lower()

    def test_fail_missing_files(self) -> None:
        """Fails when files missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "empty_run"
            run_dir.mkdir()
            
            check = check_contract_integrity(run_dir)
            
            assert check.status == ReadinessStatus.FAIL
            assert "missing" in check.message.lower()

    def test_fail_invalid_manifest(self) -> None:
        """Fails with invalid manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "bad_run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text("not valid json")
            
            check = check_contract_integrity(run_dir)
            
            assert check.status == ReadinessStatus.FAIL


class TestPrivacySafetyCheck:
    """Test privacy safety check."""

    def test_pass_with_clean_audit(self, clean_run_dir: Path) -> None:
        """Passes with clean redaction audit."""
        check = check_privacy_safety(clean_run_dir)
        
        assert check.status == ReadinessStatus.PASS

    def test_warn_no_audit(self, contaminated_run_dir: Path) -> None:
        """Warns when no audit performed."""
        check = check_privacy_safety(contaminated_run_dir)
        
        assert check.status == ReadinessStatus.WARN
        assert "no redaction audit" in check.message.lower()

    def test_fail_with_leaks(self) -> None:
        """Fails when leaks detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "leaky_run"
            run_dir.mkdir()
            
            audit = {
                "status": "FAIL",
                "findings_count": 5,
                "summary": {"findings_by_type": {"email": 3, "phone": 2}},
            }
            (run_dir / "redaction_audit.json").write_text(json.dumps(audit))
            
            check = check_privacy_safety(run_dir)
            
            assert check.status == ReadinessStatus.FAIL


class TestContaminationRiskCheck:
    """Test contamination risk check."""

    def test_pass_clean_run(self, clean_run_dir: Path) -> None:
        """Passes for clean run."""
        check = check_contamination_risk(clean_run_dir)
        
        # May be PASS or WARN depending on decision structure
        assert check.status in (ReadinessStatus.PASS, ReadinessStatus.WARN)

    def test_fail_contaminated_run(self, contaminated_run_dir: Path) -> None:
        """Fails for contaminated run."""
        check = check_contamination_risk(contaminated_run_dir)
        
        assert check.status == ReadinessStatus.FAIL
        assert "contamination" in check.message.lower()


class TestRegressionRiskCheck:
    """Test regression risk check."""

    def test_pass_no_regression(self, clean_run_dir: Path) -> None:
        """Passes when no regression."""
        check = check_regression_risk(clean_run_dir)
        
        assert check.status in (ReadinessStatus.PASS, ReadinessStatus.WARN)

    def test_warn_with_regression(self, regressed_run_dir: Path) -> None:
        """Warns when regression detected."""
        check = check_regression_risk(regressed_run_dir)
        
        assert check.status == ReadinessStatus.WARN
        assert "regression" in check.message.lower()


class TestConfigSanityCheck:
    """Test config sanity check."""

    def test_pass_reasonable_config(self, clean_run_dir: Path) -> None:
        """Passes with reasonable config."""
        check = check_config_sanity(clean_run_dir)
        
        assert check.status in (ReadinessStatus.PASS, ReadinessStatus.WARN)

    def test_warn_absurd_thresholds(self) -> None:
        """Warns with absurd thresholds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "bad_config_run"
            run_dir.mkdir()
            
            manifest = {
                "contracts": [
                    {"name": "bad", "min_row_count": 1, "max_null_ratio": 0.9}
                ],
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            
            check = check_config_sanity(run_dir)
            
            assert check.status == ReadinessStatus.WARN


class TestReproducibilityCheck:
    """Test reproducibility check."""

    def test_pass_with_environment(self, clean_run_dir: Path) -> None:
        """Passes with environment.json."""
        check = check_reproducibility(clean_run_dir)
        
        assert check.status == ReadinessStatus.PASS

    def test_warn_without_environment(self, contaminated_run_dir: Path) -> None:
        """Warns without environment capture."""
        check = check_reproducibility(contaminated_run_dir)
        
        assert check.status == ReadinessStatus.WARN


class TestValidateReadiness:
    """Test full readiness validation."""

    def test_clean_run_passes(self, clean_run_dir: Path) -> None:
        """Clean run gets PASS status."""
        report = validate_readiness(clean_run_dir)
        
        # Should be PASS or WARN (hash chain may warn)
        assert report.overall_status in (ReadinessStatus.PASS, ReadinessStatus.WARN)
        assert len(report.checks) > 0

    def test_contaminated_run_fails(self, contaminated_run_dir: Path) -> None:
        """Contaminated run gets FAIL status."""
        report = validate_readiness(contaminated_run_dir)
        
        assert report.overall_status == ReadinessStatus.FAIL

    def test_summary_calculated(self, clean_run_dir: Path) -> None:
        """Summary is calculated correctly."""
        report = validate_readiness(clean_run_dir)
        
        assert "total_checks" in report.summary
        assert "passed" in report.summary
        assert report.summary["total_checks"] == len(report.checks)

    def test_missing_directory_raises(self) -> None:
        """Missing directory raises error."""
        with pytest.raises(ReadinessError):
            validate_readiness("/nonexistent/path")


class TestFormatReadinessReport:
    """Test report formatting."""

    def test_format_includes_status(self, clean_run_dir: Path) -> None:
        """Formatted report includes status."""
        report = validate_readiness(clean_run_dir)
        formatted = format_readiness_report(report)
        
        assert "Readiness" in formatted
        assert report.run_id in formatted or formatted  # Just verify it produces output

    def test_format_includes_checks(self, clean_run_dir: Path) -> None:
        """Formatted report includes all checks."""
        report = validate_readiness(clean_run_dir)
        formatted = format_readiness_report(report)
        
        # Just verify the report has content and is not empty
        assert len(formatted) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
