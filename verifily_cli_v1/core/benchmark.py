"""Benchmark module for measuring Verifily performance.

Provides deterministic performance measurement for ingest,
contamination detection, and pipeline operations.
"""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from verifily_cli_v1.core.integrity import canonical_hash


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    name: str
    rows: int = 0
    bytes_processed: int = 0
    elapsed_ms: float = 0.0
    rows_per_sec: float = 0.0
    memory_mb: float = 0.0
    iterations: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "bytes_processed": self.bytes_processed,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "rows_per_sec": round(self.rows_per_sec, 2),
            "memory_mb": round(self.memory_mb, 2),
            "iterations": self.iterations,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report with multiple results."""
    
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class BenchmarkError(Exception):
    """Benchmark execution failed."""
    pass


def _count_rows_jsonl(file_path: Path) -> int:
    """Count rows in a JSONL file."""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
    return count


def _get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def _measure_memory_mb(func: Callable, *args, **kwargs) -> tuple:
    """Measure memory usage of a function."""
    tracemalloc.start()
    
    result = func(*args, **kwargs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Convert to MB
    peak_mb = peak / (1024 * 1024)
    
    return result, peak_mb


def benchmark_ingest(
    dataset_path: Union[str, Path],
    runs: int = 3,
) -> BenchmarkResult:
    """Benchmark dataset ingestion performance.
    
    Measures rows/second for reading and parsing JSONL files.
    
    Args:
        dataset_path: Path to JSONL dataset
        runs: Number of runs for averaging
        
    Returns:
        BenchmarkResult with performance metrics
        
    Example:
        >>> result = benchmark_ingest("data/train.jsonl", runs=3)
        >>> print(f"{result.rows_per_sec:.0f} rows/sec")
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise BenchmarkError(f"Dataset not found: {dataset_path}")
    
    # Count rows and size once
    rows = _count_rows_jsonl(dataset_path)
    file_size = _get_file_size(dataset_path)
    
    if rows == 0:
        raise BenchmarkError(f"Dataset is empty: {dataset_path}")
    
    # Benchmark: read and parse the file multiple times
    total_elapsed_ms = 0.0
    total_memory_mb = 0.0
    
    for _ in range(runs):
        tracemalloc.start()
        start = time.monotonic()
        
        # Simulate ingest: read and parse JSONL
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        pass  # Skip invalid lines in benchmark
        
        elapsed_ms = (time.monotonic() - start) * 1000
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_elapsed_ms += elapsed_ms
        total_memory_mb += peak / (1024 * 1024)
    
    avg_elapsed_ms = total_elapsed_ms / runs
    avg_memory_mb = total_memory_mb / runs
    rows_per_sec = (rows * 1000) / avg_elapsed_ms if avg_elapsed_ms > 0 else 0
    
    return BenchmarkResult(
        name="ingest",
        rows=rows,
        bytes_processed=file_size,
        elapsed_ms=avg_elapsed_ms,
        rows_per_sec=rows_per_sec,
        memory_mb=avg_memory_mb,
        iterations=runs,
    )


def benchmark_contamination(
    train_path: Union[str, Path],
    eval_path: Union[str, Path],
    runs: int = 1,
) -> BenchmarkResult:
    """Benchmark contamination detection performance.
    
    Args:
        train_path: Path to training dataset
        eval_path: Path to evaluation dataset
        runs: Number of runs for averaging
        
    Returns:
        BenchmarkResult with performance metrics
    """
    train_path = Path(train_path)
    eval_path = Path(eval_path)
    
    if not train_path.exists():
        raise BenchmarkError(f"Train dataset not found: {train_path}")
    if not eval_path.exists():
        raise BenchmarkError(f"Eval dataset not found: {eval_path}")
    
    train_rows = _count_rows_jsonl(train_path)
    eval_rows = _count_rows_jsonl(eval_path)
    total_rows = train_rows + eval_rows
    
    total_bytes = _get_file_size(train_path) + _get_file_size(eval_path)
    
    # Benchmark contamination detection (simplified)
    total_elapsed_ms = 0.0
    total_memory_mb = 0.0
    
    for _ in range(runs):
        tracemalloc.start()
        start = time.monotonic()
        
        # Simulate contamination check: compare n-grams
        train_ngrams = _extract_ngrams_sample(train_path, sample_size=1000)
        eval_ngrams = _extract_ngrams_sample(eval_path, sample_size=500)
        
        # Jaccard similarity calculation
        if train_ngrams and eval_ngrams:
            intersection = len(train_ngrams & eval_ngrams)
            union = len(train_ngrams | eval_ngrams)
            similarity = intersection / union if union > 0 else 0.0
        
        elapsed_ms = (time.monotonic() - start) * 1000
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_elapsed_ms += elapsed_ms
        total_memory_mb += peak / (1024 * 1024)
    
    avg_elapsed_ms = total_elapsed_ms / runs
    avg_memory_mb = total_memory_mb / runs
    rows_per_sec = (total_rows * 1000) / avg_elapsed_ms if avg_elapsed_ms > 0 else 0
    
    return BenchmarkResult(
        name="contamination",
        rows=total_rows,
        bytes_processed=total_bytes,
        elapsed_ms=avg_elapsed_ms,
        rows_per_sec=rows_per_sec,
        memory_mb=avg_memory_mb,
        iterations=runs,
    )


def _extract_ngrams_sample(file_path: Path, sample_size: int = 1000, n: int = 3) -> set:
    """Extract n-grams from a sample of the dataset."""
    ngrams = set()
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                text = ""
                if isinstance(obj, dict):
                    # Extract text from common fields
                    text = obj.get("text", "") or obj.get("instruction", "") or obj.get("content", "")
                    if isinstance(text, list):
                        text = " ".join(str(t) for t in text)
                
                if text:
                    # Simple word n-grams
                    words = str(text).lower().split()
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i:i+n])
                        ngrams.add(ngram)
                
                count += 1
            except (json.JSONDecodeError, AttributeError):
                continue
    
    return ngrams


def benchmark_pipeline(
    run_dir: Union[str, Path],
    plan: bool = True,
    runs: int = 1,
) -> BenchmarkResult:
    """Benchmark pipeline execution performance.
    
    Args:
        run_dir: Path to run directory with artifacts
        plan: Whether to run in plan mode
        runs: Number of runs for averaging
        
    Returns:
        BenchmarkResult with performance metrics
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        raise BenchmarkError(f"Run directory not found: {run_path}")
    
    # Find dataset
    dataset_path = run_path / "dataset.jsonl"
    if not dataset_path.exists():
        # Try to find any JSONL file
        jsonl_files = list(run_path.glob("*.jsonl"))
        if jsonl_files:
            dataset_path = jsonl_files[0]
        else:
            raise BenchmarkError(f"No dataset found in {run_path}")
    
    rows = _count_rows_jsonl(dataset_path)
    file_size = _get_file_size(dataset_path)
    
    # Benchmark pipeline simulation
    total_elapsed_ms = 0.0
    total_memory_mb = 0.0
    
    for _ in range(runs):
        tracemalloc.start()
        start = time.monotonic()
        
        # Simulate pipeline stages
        # Stage 1: Ingest
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        # Stage 2: Schema validation (simulated)
        valid_count = 0
        for record in data:
            if isinstance(record, dict):
                valid_count += 1
        
        # Stage 3: Decision (simulated)
        passed = valid_count > len(data) * 0.9
        
        elapsed_ms = (time.monotonic() - start) * 1000
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_elapsed_ms += elapsed_ms
        total_memory_mb += peak / (1024 * 1024)
    
    avg_elapsed_ms = total_elapsed_ms / runs
    avg_memory_mb = total_memory_mb / runs
    rows_per_sec = (rows * 1000) / avg_elapsed_ms if avg_elapsed_ms > 0 else 0
    
    return BenchmarkResult(
        name=f"pipeline{'_plan' if plan else ''}",
        rows=rows,
        bytes_processed=file_size,
        elapsed_ms=avg_elapsed_ms,
        rows_per_sec=rows_per_sec,
        memory_mb=avg_memory_mb,
        iterations=runs,
    )


def run_benchmark_suite(
    dataset_path: Optional[Union[str, Path]] = None,
    train_path: Optional[Union[str, Path]] = None,
    eval_path: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
) -> BenchmarkReport:
    """Run a complete benchmark suite.
    
    Args:
        dataset_path: Path to dataset for ingest benchmark
        train_path: Path to training dataset for contamination
        eval_path: Path to evaluation dataset for contamination
        run_dir: Path to run directory for pipeline benchmark
        
    Returns:
        BenchmarkReport with all results
    """
    report = BenchmarkReport()
    
    # Ingest benchmark
    if dataset_path and Path(dataset_path).exists():
        try:
            result = benchmark_ingest(dataset_path, runs=3)
            report.add_result(result)
        except BenchmarkError as e:
            report.add_result(BenchmarkResult(
                name="ingest",
                rows=0,
                elapsed_ms=0,
                rows_per_sec=0,
            ))
    
    # Contamination benchmark
    if train_path and eval_path and Path(train_path).exists() and Path(eval_path).exists():
        try:
            result = benchmark_contamination(train_path, eval_path, runs=1)
            report.add_result(result)
        except BenchmarkError as e:
            report.add_result(BenchmarkResult(
                name="contamination",
                rows=0,
                elapsed_ms=0,
                rows_per_sec=0,
            ))
    
    # Pipeline benchmark
    if run_dir and Path(run_dir).exists():
        try:
            result = benchmark_pipeline(run_dir, plan=True, runs=1)
            report.add_result(result)
        except BenchmarkError as e:
            report.add_result(BenchmarkResult(
                name="pipeline_plan",
                rows=0,
                elapsed_ms=0,
                rows_per_sec=0,
            ))
    
    # Calculate summary
    if report.results:
        total_rows = sum(r.rows for r in report.results)
        total_time = sum(r.elapsed_ms for r in report.results)
        avg_rows_per_sec = sum(r.rows_per_sec for r in report.results) / len(report.results)
        
        report.summary = {
            "total_benchmarks": len(report.results),
            "total_rows_processed": total_rows,
            "total_time_ms": round(total_time, 2),
            "avg_rows_per_sec": round(avg_rows_per_sec, 2),
        }
    
    return report


def format_benchmark_table(report: BenchmarkReport) -> str:
    """Format benchmark report as a rich table."""
    lines = []
    lines.append("Benchmark Results")
    lines.append("=" * 70)
    lines.append(f"{'Name':<20} {'Rows':<10} {'Time (ms)':<12} {'Rows/sec':<12} {'Memory (MB)':<12}")
    lines.append("-" * 70)
    
    for result in report.results:
        lines.append(
            f"{result.name:<20} "
            f"{result.rows:<10,} "
            f"{result.elapsed_ms:<12.1f} "
            f"{result.rows_per_sec:<12,.0f} "
            f"{result.memory_mb:<12.1f}"
        )
    
    lines.append("-" * 70)
    
    if report.summary:
        lines.append(f"\nSummary:")
        lines.append(f"  Total benchmarks: {report.summary.get('total_benchmarks', 0)}")
        lines.append(f"  Total rows: {report.summary.get('total_rows_processed', 0):,}")
        lines.append(f"  Avg rows/sec: {report.summary.get('avg_rows_per_sec', 0):,.0f}")
    
    return "\n".join(lines)
