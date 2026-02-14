"""Tests for benchmark module.

Target: ~10 tests, runtime <1.0s (loose thresholds for CI stability)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from verifily_cli_v1.core.benchmark import (
    BenchmarkResult,
    BenchmarkReport,
    benchmark_ingest,
    benchmark_contamination,
    benchmark_pipeline,
    run_benchmark_suite,
    format_benchmark_table,
    BenchmarkError,
)


@pytest.fixture
def small_dataset() -> Generator[Path, None, None]:
    """Create a small test dataset."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(100):
            record = {
                "id": i,
                "text": f"Sample text for record {i}",
                "label": "positive" if i % 2 == 0 else "negative",
            }
            f.write(json.dumps(record) + '\n')
        f.flush()
        yield Path(f.name)


@pytest.fixture
def tiny_datasets() -> Generator[tuple[Path, Path], None, None]:
    """Create tiny train/eval datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.jsonl"
        eval_path = Path(tmpdir) / "eval.jsonl"
        
        # Train: 50 rows
        with open(train_path, 'w') as f:
            for i in range(50):
                record = {"instruction": f"Train task {i}", "output": f"Train output {i}"}
                f.write(json.dumps(record) + '\n')
        
        # Eval: 30 rows
        with open(eval_path, 'w') as f:
            for i in range(30):
                record = {"instruction": f"Eval task {i}", "output": f"Eval output {i}"}
                f.write(json.dumps(record) + '\n')
        
        yield train_path, eval_path


@pytest.fixture
def mock_run_dir() -> Generator[Path, None, None]:
    """Create a mock run directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_001"
        run_dir.mkdir()
        
        # Create dataset
        dataset_path = run_dir / "dataset.jsonl"
        with open(dataset_path, 'w') as f:
            for i in range(50):
                record = {"text": f"Record {i}", "label": "A"}
                f.write(json.dumps(record) + '\n')
        
        # Create manifest
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"run_id": "run_001", "status": "complete"}))
        
        yield run_dir


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_result_creation(self) -> None:
        """Can create benchmark result."""
        result = BenchmarkResult(
            name="test",
            rows=1000,
            elapsed_ms=100.0,
            rows_per_sec=10000.0,
        )
        
        assert result.name == "test"
        assert result.rows == 1000
        assert result.elapsed_ms == 100.0
        assert result.rows_per_sec == 10000.0

    def test_result_to_dict(self) -> None:
        """Result converts to dict."""
        result = BenchmarkResult(
            name="ingest",
            rows=1000,
            bytes_processed=50000,
            elapsed_ms=100.0,
            rows_per_sec=10000.0,
            memory_mb=50.0,
            iterations=3,
        )
        
        d = result.to_dict()
        assert d["name"] == "ingest"
        assert d["rows"] == 1000
        assert "rows_per_sec" in d


class TestBenchmarkIngest:
    """Test ingest benchmark."""

    def test_ingest_completes(self, small_dataset: Path) -> None:
        """Ingest benchmark completes successfully."""
        result = benchmark_ingest(small_dataset, runs=1)
        
        assert result.name == "ingest"
        assert result.rows == 100
        assert result.elapsed_ms > 0
        assert result.rows_per_sec > 0

    def test_ingest_multiple_runs(self, small_dataset: Path) -> None:
        """Ingest benchmark averages multiple runs."""
        result = benchmark_ingest(small_dataset, runs=3)
        
        assert result.iterations == 3
        assert result.rows == 100

    def test_ingest_empty_file_raises(self) -> None:
        """Empty dataset raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("")  # Empty file
            f.flush()
            empty_path = Path(f.name)
        
        with pytest.raises(BenchmarkError):
            benchmark_ingest(empty_path, runs=1)

    def test_ingest_missing_file_raises(self) -> None:
        """Missing file raises error."""
        with pytest.raises(BenchmarkError):
            benchmark_ingest("/nonexistent/path.jsonl", runs=1)

    def test_ingest_deterministic_timing(self, small_dataset: Path) -> None:
        """Timing is in reasonable bounds (very loose for CI)."""
        result = benchmark_ingest(small_dataset, runs=1)
        
        # Should be reasonably fast (under 5 seconds for 100 rows)
        assert result.elapsed_ms < 5000
        # Should have positive rows/sec
        assert result.rows_per_sec > 0


class TestBenchmarkContamination:
    """Test contamination benchmark."""

    def test_contamination_completes(self, tiny_datasets: tuple[Path, Path]) -> None:
        """Contamination benchmark completes."""
        train_path, eval_path = tiny_datasets
        result = benchmark_contamination(train_path, eval_path, runs=1)
        
        assert result.name == "contamination"
        assert result.rows == 80  # 50 + 30
        assert result.elapsed_ms > 0

    def test_contamination_missing_files_raises(self) -> None:
        """Missing files raise error."""
        with pytest.raises(BenchmarkError):
            benchmark_contamination("/nonexistent/train.jsonl", "/nonexistent/eval.jsonl")


class TestBenchmarkPipeline:
    """Test pipeline benchmark."""

    def test_pipeline_completes(self, mock_run_dir: Path) -> None:
        """Pipeline benchmark completes."""
        result = benchmark_pipeline(mock_run_dir, plan=True, runs=1)
        
        assert "pipeline" in result.name
        assert result.rows == 50
        assert result.elapsed_ms > 0
        assert result.rows_per_sec > 0

    def test_pipeline_missing_directory_raises(self) -> None:
        """Missing directory raises error."""
        with pytest.raises(BenchmarkError):
            benchmark_pipeline("/nonexistent/run_dir", plan=True)


class TestBenchmarkSuite:
    """Test full benchmark suite."""

    def test_suite_runs_all_benchmarks(self, small_dataset: Path, tiny_datasets: tuple[Path, Path], mock_run_dir: Path) -> None:
        """Suite runs all available benchmarks."""
        train_path, eval_path = tiny_datasets
        
        report = run_benchmark_suite(
            dataset_path=small_dataset,
            train_path=train_path,
            eval_path=eval_path,
            run_dir=mock_run_dir,
        )
        
        assert len(report.results) == 3
        assert report.summary["total_benchmarks"] == 3
        assert report.summary["total_rows_processed"] > 0

    def test_suite_partial_benchmarks(self, small_dataset: Path) -> None:
        """Suite works with partial benchmarks."""
        report = run_benchmark_suite(dataset_path=small_dataset)
        
        assert len(report.results) == 1
        assert report.results[0].name == "ingest"

    def test_suite_empty_benchmarks(self) -> None:
        """Suite handles no benchmarks gracefully."""
        report = run_benchmark_suite()
        
        assert len(report.results) == 0


class TestBenchmarkOutput:
    """Test benchmark output formatting."""

    def test_format_table(self, small_dataset: Path) -> None:
        """Table formatting works."""
        result = benchmark_ingest(small_dataset, runs=1)
        report = BenchmarkReport(results=[result])
        
        table = format_benchmark_table(report)
        
        assert "Benchmark Results" in table
        assert "ingest" in table
        assert str(result.rows) in table

    def test_report_to_json(self, small_dataset: Path) -> None:
        """Report converts to JSON."""
        result = benchmark_ingest(small_dataset, runs=1)
        report = BenchmarkReport(results=[result])
        report.summary = {"total_benchmarks": 1}
        
        json_str = report.to_json()
        
        parsed = json.loads(json_str)
        assert parsed["summary"]["total_benchmarks"] == 1


class TestBenchmarkGuardrails:
    """Test performance guardrails."""

    def test_rows_per_sec_threshold(self, small_dataset: Path) -> None:
        """Rows/sec stays above minimal threshold (very low for CI)."""
        result = benchmark_ingest(small_dataset, runs=1)
        
        # Minimal threshold: 100 rows/sec (very low, should always pass)
        assert result.rows_per_sec > 100, f"Performance too slow: {result.rows_per_sec} rows/sec"

    def test_memory_usage_reasonable(self, small_dataset: Path) -> None:
        """Memory usage is reasonable (under 100MB for small dataset)."""
        result = benchmark_ingest(small_dataset, runs=1)
        
        # Should use less than 100MB for 100 rows
        assert result.memory_mb < 100, f"Memory usage too high: {result.memory_mb} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
