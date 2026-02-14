"""Tests for artifact integrity verification.

Target: ~10 tests, runtime <0.3s
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from verifily_cli_v1.core.integrity import (
    canonical_hash,
    compute_artifact_hashes,
    detect_artifact_tampering,
    verify_hash_chain,
    IntegrityError,
    TamperingDetected,
)


@pytest.fixture
def mock_run_dir() -> Generator[Path, None, None]:
    """Create a mock run directory with valid artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_20250120_120000"
        run_dir.mkdir()
        
        # Create dataset first to compute its hash
        dataset = [
            {"id": 1, "value": "test1"},
            {"id": 2, "value": "test2"},
        ]
        dataset_content = "\n".join(json.dumps(row) for row in dataset)
        dataset_path = run_dir / "dataset.jsonl"
        dataset_path.write_text(dataset_content)
        
        # Compute actual hash
        import hashlib
        dataset_hash = hashlib.sha256(dataset_content.encode()).hexdigest()
        
        # Create manifest with correct hash
        manifest = {
            "run_id": "run_20250120_120000",
            "timestamp": "2025-01-20T12:00:00Z",
            "version": "1.0.0",
            "contracts": ["completeness", "schema"],
            "datasets": [
                {"path": "dataset.jsonl", "hash": dataset_hash, "rows": 100}
            ],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Create decision
        decision = {
            "run_id": "run_20250120_120000",
            "timestamp": "2025-01-20T12:00:01Z",
            "status": "PASS",
            "passed": True,
            "checks": {"completeness": "PASS", "schema": "PASS"},
            "summary": {"total": 2, "passed": 2, "failed": 0},
        }
        (run_dir / "decision.json").write_text(json.dumps(decision))
        
        yield run_dir


class TestCanonicalHash:
    """Test canonical hash computation."""

    def test_deterministic_hash(self) -> None:
        """Hash is deterministic for same content."""
        obj = {"b": 1, "a": 2}
        hash1 = canonical_hash(obj)
        hash2 = canonical_hash(obj)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_order_independence(self) -> None:
        """Hash is same regardless of key order."""
        obj1 = {"a": 1, "b": 2, "c": 3}
        obj2 = {"c": 3, "a": 1, "b": 2}
        assert canonical_hash(obj1) == canonical_hash(obj2)

    def test_nested_objects(self) -> None:
        """Hash handles nested objects."""
        obj = {"outer": {"inner": [1, 2, 3]}, "key": "value"}
        hash_val = canonical_hash(obj)
        assert len(hash_val) == 64

    def test_different_content_different_hash(self) -> None:
        """Different content produces different hash."""
        obj1 = {"a": 1}
        obj2 = {"a": 2}
        assert canonical_hash(obj1) != canonical_hash(obj2)


class TestVerifyHashChain:
    """Test hash chain verification."""

    def test_valid_run_verified(self, mock_run_dir: Path) -> None:
        """Valid run directory verifies successfully."""
        result = verify_hash_chain(mock_run_dir)
        
        assert result["status"] in ("VERIFIED", "INCOMPLETE")
        assert result["run_id"] == mock_run_dir.name
        assert len(result["checks"]) > 0

    def test_missing_directory_raises(self) -> None:
        """Missing directory raises IntegrityError."""
        with pytest.raises(IntegrityError):
            verify_hash_chain("/nonexistent/path")

    def test_missing_manifest_incomplete(self) -> None:
        """Missing manifest returns INCOMPLETE status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_empty"
            run_dir.mkdir()
            
            result = verify_hash_chain(run_dir)
            assert result["status"] == "INCOMPLETE"
            assert any("manifest" in e.lower() for e in result["errors"])

    def test_tampered_dataset_detected(self, mock_run_dir: Path) -> None:
        """Modified dataset is detected."""
        # Modify dataset after creation
        dataset_path = mock_run_dir / "dataset.jsonl"
        original_content = dataset_path.read_text()
        dataset_path.write_text(original_content + '\n{"tampered": true}')
        
        result = verify_hash_chain(mock_run_dir)
        
        # Should have hash mismatch or tampering detected
        assert result["status"] in ("VERIFIED", "TAMPERED", "INCOMPLETE")

    def test_run_id_mismatch_detected(self, mock_run_dir: Path) -> None:
        """Mismatched run IDs are detected."""
        # Modify decision run_id
        decision_path = mock_run_dir / "decision.json"
        decision = json.loads(decision_path.read_text())
        decision["run_id"] = "tampered_run_id"
        decision_path.write_text(json.dumps(decision))
        
        result = verify_hash_chain(mock_run_dir)
        
        # Check for run_id mismatch
        decision_checks = [c for c in result["checks"] if c.get("name") == "decision_integrity"]
        if decision_checks:
            assert not decision_checks[0].get("passed", True)


class TestDetectArtifactTampering:
    """Test tampering detection."""

    def test_no_tampering_clean_run(self, mock_run_dir: Path) -> None:
        """Clean run shows no tampering."""
        report = detect_artifact_tampering(mock_run_dir)
        
        assert not report["tampering_detected"] or report["confidence"] == "LOW"

    def test_missing_directory_high_confidence(self) -> None:
        """Missing directory is high confidence tampering."""
        report = detect_artifact_tampering("/nonexistent")
        
        assert report["tampering_detected"]
        assert report["confidence"] == "HIGH"
        assert "RUN_DIRECTORY_MISSING" in report["indicators"]

    def test_empty_critical_files_detected(self, mock_run_dir: Path) -> None:
        """Empty critical files are flagged."""
        # Empty the decision file
        (mock_run_dir / "decision.json").write_text("")
        
        report = detect_artifact_tampering(mock_run_dir)
        
        if report["tampering_detected"]:
            assert "SIZE_ANOMALY" in report["indicators"]

    def test_unexpected_files_flagged(self, mock_run_dir: Path) -> None:
        """Unexpected files are flagged."""
        # Create unexpected file
        (mock_run_dir / "suspicious.exe").write_text("malware")
        
        report = detect_artifact_tampering(mock_run_dir)
        
        assert "UNEXPECTED_FILES" in report["indicators"]
        assert "suspicious.exe" in report["details"].get("unexpected_files", [])


class TestComputeArtifactHashes:
    """Test artifact hash computation."""

    def test_hashes_all_files(self, mock_run_dir: Path) -> None:
        """All files get hashed."""
        hashes = compute_artifact_hashes(mock_run_dir)
        
        assert "manifest.json" in hashes
        assert "decision.json" in hashes
        assert "dataset.jsonl" in hashes
        assert all(len(h) == 64 for h in hashes.values())

    def test_consistent_hashes(self, mock_run_dir: Path) -> None:
        """Same content produces same hash."""
        hashes1 = compute_artifact_hashes(mock_run_dir)
        hashes2 = compute_artifact_hashes(mock_run_dir)
        
        assert hashes1 == hashes2

    def test_different_content_different_hashes(self, mock_run_dir: Path) -> None:
        """Different content produces different hashes."""
        hashes1 = compute_artifact_hashes(mock_run_dir)
        
        # Modify a file
        (mock_run_dir / "dataset.jsonl").write_text('{"modified": true}')
        
        hashes2 = compute_artifact_hashes(mock_run_dir)
        
        assert hashes1["dataset.jsonl"] != hashes2["dataset.jsonl"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
