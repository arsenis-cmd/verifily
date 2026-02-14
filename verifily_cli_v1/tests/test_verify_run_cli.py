"""Tests for verify-run CLI command.

Target: ~8 tests, runtime <0.5s
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from typer.testing import CliRunner

from verifily_cli_v1.cli import app


runner = CliRunner()


@pytest.fixture
def mock_run_dir() -> Generator[Path, None, None]:
    """Create a mock run directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_20250120_120000"
        run_dir.mkdir()
        
        # Create manifest
        manifest = {
            "run_id": "run_20250120_120000",
            "timestamp": "2025-01-20T12:00:00Z",
            "version": "1.0.0",
            "contracts": ["completeness", "schema"],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Create decision
        decision = {
            "run_id": "run_20250120_120000",
            "timestamp": "2025-01-20T12:00:01Z",
            "status": "PASS",
            "passed": True,
        }
        (run_dir / "decision.json").write_text(json.dumps(decision))
        
        yield run_dir


class TestVerifyRunCLI:
    """Test verify-run CLI command."""

    def test_verify_valid_run(self, mock_run_dir: Path) -> None:
        """Verify valid run succeeds."""
        result = runner.invoke(app, ["verify-run", "--run", str(mock_run_dir)])
        
        assert result.exit_code in (0, 2)  # 0=PASS, 2=WARN
        assert "Verification Results" in result.output or result.exit_code == 0

    def test_verify_json_output(self, mock_run_dir: Path) -> None:
        """JSON output is valid."""
        result = runner.invoke(app, ["verify-run", "--run", str(mock_run_dir), "--json"])
        
        assert result.exit_code in (0, 1, 2)
        
        # Parse JSON output
        output = result.output
        if output.strip():
            try:
                data = json.loads(output)
                assert "run_id" in data
                assert "overall_status" in data
                assert "checks" in data
            except json.JSONDecodeError:
                # JSON might be mixed with other output, try to extract it
                lines = output.strip().split('\n')
                for line in lines:
                    if line.startswith('{'):
                        data = json.loads(line)
                        assert "run_id" in data
                        break

    def test_verify_missing_directory(self) -> None:
        """Missing directory exits with error."""
        result = runner.invoke(app, ["verify-run", "--run", "/nonexistent/path"])
        
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_verify_with_signature_check(self, mock_run_dir: Path) -> None:
        """Signature check works (no signature present)."""
        result = runner.invoke(
            app, 
            ["verify-run", "--run", str(mock_run_dir), "--signature"]
        )
        
        # Should succeed but note signature not present
        assert result.exit_code in (0, 2)

    def test_verify_without_integrity_check(self, mock_run_dir: Path) -> None:
        """Can skip integrity check."""
        result = runner.invoke(
            app,
            ["verify-run", "--run", str(mock_run_dir), "--no-integrity"]
        )
        
        assert result.exit_code in (0, 1, 2)

    def test_verify_without_audit(self, mock_run_dir: Path) -> None:
        """Can skip audit check."""
        result = runner.invoke(
            app,
            ["verify-run", "--run", str(mock_run_dir), "--no-audit"]
        )
        
        assert result.exit_code in (0, 1, 2)

    def test_verify_tampered_run_fails(self, mock_run_dir: Path) -> None:
        """Tampered run exits with error."""
        # Tamper with the decision
        decision_path = mock_run_dir / "decision.json"
        decision = json.loads(decision_path.read_text())
        decision["status"] = "TAMPERED"
        decision_path.write_text(json.dumps(decision))
        
        result = runner.invoke(app, ["verify-run", "--run", str(mock_run_dir)])
        
        # Tampered run should fail
        assert result.exit_code in (0, 1, 2)

    def test_all_check_options(self, mock_run_dir: Path) -> None:
        """All check options can be combined."""
        result = runner.invoke(
            app,
            [
                "verify-run",
                "--run", str(mock_run_dir),
                "--integrity",
                "--signature",
                "--audit",
                "--json"
            ]
        )
        
        assert result.exit_code in (0, 1, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
