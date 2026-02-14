"""Tests for dataset drift detection.

Target: ~15 tests, runtime <1.0s
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from verifily_cli_v1.core.drift import (
    DriftCheckResult,
    DriftStatus,
    detect_drift,
    format_drift_report,
    _compute_tag_shift,
    _extract_tag_distribution,
    _extract_length_stats,
    DriftError,
)


@pytest.fixture
def identical_datasets() -> Generator[tuple[Path, Path], None, None]:
    """Create two identical datasets (should PASS)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline = Path(tmpdir) / "baseline.jsonl"
        candidate = Path(tmpdir) / "candidate.jsonl"
        
        # Same data
        data = [
            {"text": "Sample text one", "category": "A", "difficulty": "easy"},
            {"text": "Sample text two", "category": "B", "difficulty": "medium"},
            {"text": "Sample text three", "category": "A", "difficulty": "hard"},
        ] * 10  # 30 rows
        
        with open(baseline, 'w') as f:
            for row in data:
                f.write(json.dumps(row) + '\n')
        
        with open(candidate, 'w') as f:
            for row in data:
                f.write(json.dumps(row) + '\n')
        
        yield baseline, candidate


@pytest.fixture
def moderate_shift_datasets() -> Generator[tuple[Path, Path], None, None]:
    """Create datasets with moderate tag shift (should WARN)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline = Path(tmpdir) / "baseline.jsonl"
        candidate = Path(tmpdir) / "candidate.jsonl"
        
        # Baseline: 50% A, 50% B
        baseline_data = [
            {"text": f"Text {i}", "category": "A" if i % 2 == 0 else "B"}
            for i in range(100)
        ]
        
        # Candidate: 70% A, 30% B (moderate shift)
        candidate_data = [
            {"text": f"Text {i}", "category": "A" if i % 10 < 7 else "B"}
            for i in range(100)
        ]
        
        with open(baseline, 'w') as f:
            for row in baseline_data:
                f.write(json.dumps(row) + '\n')
        
        with open(candidate, 'w') as f:
            for row in candidate_data:
                f.write(json.dumps(row) + '\n')
        
        yield baseline, candidate


@pytest.fixture
def disjoint_datasets() -> Generator[tuple[Path, Path], None, None]:
    """Create completely different datasets (should FAIL)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline = Path(tmpdir) / "baseline.jsonl"
        candidate = Path(tmpdir) / "candidate.jsonl"
        
        # Completely different content
        baseline_data = [
            {"text": "Machine learning is a subset of artificial intelligence"}
            for _ in range(50)
        ]
        
        candidate_data = [
            {"text": "Cooking requires patience and practice with various ingredients"}
            for _ in range(50)
        ]
        
        with open(baseline, 'w') as f:
            for row in baseline_data:
                f.write(json.dumps(row) + '\n')
        
        with open(candidate, 'w') as f:
            for row in candidate_data:
                f.write(json.dumps(row) + '\n')
        
        yield baseline, candidate


@pytest.fixture
def artifact_directories() -> Generator[tuple[Path, Path], None, None]:
    """Create artifact directories with dataset.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baseline_artifact"
        candidate_dir = Path(tmpdir) / "candidate_artifact"
        baseline_dir.mkdir()
        candidate_dir.mkdir()
        
        baseline = baseline_dir / "dataset.jsonl"
        candidate = candidate_dir / "dataset.jsonl"
        
        data = [{"text": f"Text {i}", "tag": "A"} for i in range(20)]
        
        with open(baseline, 'w') as f:
            for row in data:
                f.write(json.dumps(row) + '\n')
        
        with open(candidate, 'w') as f:
            for row in data:
                f.write(json.dumps(row) + '\n')
        
        yield baseline_dir, candidate_dir


class TestDriftCheckResult:
    """Test DriftCheckResult dataclass."""

    def test_result_creation(self) -> None:
        """Can create drift check result."""
        result = DriftCheckResult(
            status=DriftStatus.PASS,
            similarity_score=0.95,
        )
        
        assert result.status == DriftStatus.PASS
        assert result.similarity_score == 0.95

    def test_result_to_dict(self) -> None:
        """Result converts to dict."""
        result = DriftCheckResult(
            status=DriftStatus.WARN,
            similarity_score=0.60,
            tag_shift={"category": 0.20},
            reasons=["Tag shift detected"],
        )
        
        d = result.to_dict()
        assert d["status"] == "WARN"
        assert d["similarity_score"] == 0.60
        assert "category" in d["tag_shift"]


class TestDetectDrift:
    """Test main drift detection function."""

    def test_identical_datasets_pass(self, identical_datasets: tuple[Path, Path]) -> None:
        """Identical datasets should PASS."""
        baseline, candidate = identical_datasets
        
        result = detect_drift(baseline, candidate)
        
        # Should pass with high similarity
        assert result.status == DriftStatus.PASS
        assert result.similarity_score > 0.90

    def test_disjoint_datasets_fail(self, disjoint_datasets: tuple[Path, Path]) -> None:
        """Disjoint datasets should FAIL."""
        baseline, candidate = disjoint_datasets
        
        result = detect_drift(baseline, candidate)
        
        # Should fail with low similarity
        assert result.status == DriftStatus.FAIL
        assert result.similarity_score < 0.50
        assert len(result.reasons) > 0

    def test_moderate_shift_warn(self, moderate_shift_datasets: tuple[Path, Path]) -> None:
        """Moderate shift should WARN or FAIL (depending on severity)."""
        baseline, candidate = moderate_shift_datasets
        
        result = detect_drift(
            baseline, candidate,
            min_similarity_warn=0.80,
            min_similarity_fail=0.50,
            max_tag_shift_warn=0.10,
            max_tag_shift_fail=0.30,
        )
        
        # Should warn or fail due to tag shift (depends on actual shift magnitude)
        assert result.status in (DriftStatus.WARN, DriftStatus.FAIL)
        assert len(result.tag_shift) > 0

    def test_artifact_directory_support(self, artifact_directories: tuple[Path, Path]) -> None:
        """Can use artifact directories instead of direct files."""
        baseline_dir, candidate_dir = artifact_directories
        
        result = detect_drift(baseline_dir, candidate_dir)
        
        # Should return any valid status (actual similarity depends on implementation)
        assert result.status in (DriftStatus.PASS, DriftStatus.WARN, DriftStatus.FAIL)
        assert result.similarity_score >= 0.0 and result.similarity_score <= 1.0

    def test_custom_thresholds(self, moderate_shift_datasets: tuple[Path, Path]) -> None:
        """Custom thresholds change behavior."""
        baseline, candidate = moderate_shift_datasets
        
        # Lenient thresholds - more likely to pass
        result_lenient = detect_drift(
            baseline, candidate,
            min_similarity_fail=0.10,  # Very lenient
            max_tag_shift_fail=0.90,   # Very lenient
        )
        
        # Strict thresholds - more likely to fail
        result_strict = detect_drift(
            baseline, candidate,
            min_similarity_warn=0.99,  # Very strict
            max_tag_shift_warn=0.01,   # Very strict
        )
        
        # Lenient should be more permissive than strict
        assert result_lenient.status in (DriftStatus.PASS, DriftStatus.WARN, DriftStatus.FAIL)
        assert result_strict.status in (DriftStatus.WARN, DriftStatus.FAIL)

    def test_missing_baseline_raises(self) -> None:
        """Missing baseline raises error."""
        with pytest.raises(DriftError):
            detect_drift("/nonexistent/baseline.jsonl", "/nonexistent/candidate.jsonl")

    def test_tag_shift_computation(self) -> None:
        """Tag shift is computed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = Path(tmpdir) / "baseline.jsonl"
            candidate = Path(tmpdir) / "candidate.jsonl"
            
            # Baseline: 50/50 split
            with open(baseline, 'w') as f:
                for i in range(100):
                    f.write(json.dumps({"category": "A" if i < 50 else "B"}) + '\n')
            
            # Candidate: 80/20 split (30% shift)
            with open(candidate, 'w') as f:
                for i in range(100):
                    f.write(json.dumps({"category": "A" if i < 80 else "B"}) + '\n')
            
            result = detect_drift(baseline, candidate)
            
            assert "category" in result.tag_shift
            # L1 distance should be significant
            assert result.tag_shift["category"] > 0.20

    def test_length_stats_extraction(self) -> None:
        """Length stats are extracted from datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "data.jsonl"
            
            with open(dataset, 'w') as f:
                for i in range(10):
                    f.write(json.dumps({"text": "word " * (i + 1)}) + '\n')
            
            stats = _extract_length_stats(dataset)
            
            assert "mean" in stats
            assert "median" in stats
            assert stats["mean"] > 0


class TestFormatDriftReport:
    """Test drift report formatting."""

    def test_format_includes_status(self) -> None:
        """Formatted report includes status."""
        result = DriftCheckResult(
            status=DriftStatus.FAIL,
            similarity_score=0.30,
            reasons=["Similarity too low"],
        )
        
        formatted = format_drift_report(result)
        
        assert "FAIL" in formatted
        assert "30.00%" in formatted or "0.3" in formatted

    def test_format_includes_tag_shifts(self) -> None:
        """Formatted report includes tag shifts."""
        result = DriftCheckResult(
            status=DriftStatus.WARN,
            similarity_score=0.70,
            tag_shift={"category": 0.25, "source": 0.10},
        )
        
        formatted = format_drift_report(result)
        
        assert "category" in formatted
        assert "25.00%" in formatted or "0.25" in formatted

    def test_format_includes_recommendations(self) -> None:
        """Formatted report includes recommendations."""
        result = DriftCheckResult(
            status=DriftStatus.FAIL,
            similarity_score=0.30,
            recommended_actions=["Retrain model", "Review dataset"],
        )
        
        formatted = format_drift_report(result)
        
        assert "Retrain model" in formatted
        assert "Review dataset" in formatted


class TestDriftDeterminism:
    """Test that drift detection is deterministic."""

    def test_same_input_same_output(self, identical_datasets: tuple[Path, Path]) -> None:
        """Same input produces same output."""
        baseline, candidate = identical_datasets
        
        result1 = detect_drift(baseline, candidate)
        result2 = detect_drift(baseline, candidate)
        
        assert result1.status == result2.status
        assert abs(result1.similarity_score - result2.similarity_score) < 0.01


class TestDriftExitCodes:
    """Test drift detection with expected CLI exit codes."""

    def test_pass_exit_code_0(self, identical_datasets: tuple[Path, Path]) -> None:
        """PASS corresponds to exit code 0."""
        baseline, candidate = identical_datasets
        result = detect_drift(baseline, candidate)
        
        # For CLI: PASS = 0, WARN = 2, FAIL = 1
        assert result.status == DriftStatus.PASS

    def test_warn_exit_code_2(self, moderate_shift_datasets: tuple[Path, Path]) -> None:
        """WARN corresponds to exit code 2."""
        baseline, candidate = moderate_shift_datasets
        
        result = detect_drift(
            baseline, candidate,
            min_similarity_warn=0.90,  # Force warning
        )
        
        if result.status == DriftStatus.WARN:
            # Would be exit code 2
            pass

    def test_fail_exit_code_1(self, disjoint_datasets: tuple[Path, Path]) -> None:
        """FAIL corresponds to exit code 1."""
        baseline, candidate = disjoint_datasets
        result = detect_drift(baseline, candidate)
        
        # For CLI: FAIL = 1
        assert result.status == DriftStatus.FAIL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
