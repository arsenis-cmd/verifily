# Performance Guide

This guide explains how to measure and interpret Verifily's performance characteristics.

## Philosophy

**Verifily prioritizes determinism and accuracy over raw speed.**

While we optimize for reasonable performance, our primary goals are:

1. **Deterministic Results**: Same input always produces same output
2. **Correctness**: No shortcuts that compromise validation accuracy
3. **Predictability**: Performance scales linearly with data size
4. **Reliability**: Consistent behavior across runs and environments

## What We Measure

### Ingest Performance

Measures rows/second for reading and parsing JSONL datasets.

```bash
verifily benchmark --dataset data/train.jsonl
```

Output:
```
Name                 Rows       Time (ms)    Rows/sec     Memory (MB) 
ingest               10,000     245.0        40,816       12.5
```

### Contamination Detection

Measures train/eval overlap detection speed.

```bash
verifily benchmark --train data/train.jsonl --eval data/eval.jsonl
```

Uses:
- N-gram sampling for efficiency
- Jaccard similarity calculation
- Memory-efficient streaming

### Pipeline Execution

Measures full validation pipeline speed.

```bash
verifily benchmark --pipeline ./runs/run_001
```

Includes:
- Data ingestion
- Schema validation
- Contract checks
- Decision generation

## Interpreting Results

### Rows Per Second

| Speed | Assessment | Notes |
|-------|------------|-------|
| >10K rows/sec | Excellent | SSD, modern CPU |
| 5K-10K rows/sec | Good | Standard cloud instance |
| 1K-5K rows/sec | Moderate | Older hardware, complex schema |
| <1K rows/sec | Slow | Consider optimization or smaller batches |

### Memory Usage

Memory scales with:
- Dataset size
- Schema complexity
- Number of contracts

Typical memory usage:
- 1K rows: ~5-10 MB
- 10K rows: ~20-50 MB
- 100K rows: ~100-200 MB
- 1M rows: ~500MB-1GB

### Why Rows/sec Varies

Factors affecting performance:

1. **Hardware**
   - SSD vs HDD (10x difference)
   - CPU cores (parallel validation)
   - Available RAM

2. **Data Characteristics**
   - Row size (larger = slower)
   - JSON complexity (nested = slower)
   - Compression (compressed = faster I/O, slower CPU)

3. **Validation Complexity**
   - Number of contracts
   - Custom validation rules
   - External API calls (avoid in validation)

4. **Environment**
   - Concurrent workloads
   - Network latency (if using remote storage)
   - Docker/container overhead

## Benchmarking

### Quick Benchmark

```bash
# Generate test dataset
python examples/benchmark_demo/generate_large_dataset.py \
    --output data/benchmark.jsonl \
    --rows 10000 \
    --schema sft

# Run benchmark
verifily benchmark --dataset data/benchmark.jsonl --runs 3
```

### Full Suite

```bash
# Run all benchmarks
verifily benchmark \
    --dataset data/train.jsonl \
    --train data/train.jsonl \
    --eval data/eval.jsonl \
    --pipeline ./runs/latest
```

### Programmatic Benchmarking

```python
from verifily_cli_v1.core.benchmark import (
    benchmark_ingest,
    benchmark_contamination,
    run_benchmark_suite,
)

# Single benchmark
result = benchmark_ingest("data/train.jsonl", runs=3)
print(f"{result.rows_per_sec:.0f} rows/sec")

# Full suite
report = run_benchmark_suite(
    dataset_path="data/train.jsonl",
    train_path="data/train.jsonl",
    eval_path="data/eval.jsonl",
)
print(report.to_json())
```

## Performance Guardrails

Verifily includes performance tests to catch regressions:

```python
# test_benchmark.py
def test_rows_per_sec_threshold():
    result = benchmark_ingest(small_dataset, runs=1)
    assert result.rows_per_sec > 100  # Minimal threshold
```

These are loose thresholds designed to catch major regressions, not measure absolute performance.

## Optimization Tips

### For Large Datasets (>100K rows)

1. **Use Sampling**
   ```yaml
   contracts:
     - name: schema
       sample: 10000  # Validate subset
   ```

2. **Parallel Validation**
   ```yaml
   contracts:
     - name: completeness
       parallel: true
   ```

3. **Incremental Validation**
   ```bash
   # Only validate new/changed rows
   verifily pipeline --incremental
   ```

4. **Streaming Mode**
   ```yaml
   pipeline:
     streaming: true  # Don't load all data into memory
   ```

### For CI/CD

1. **Use Smaller Test Datasets**
   ```bash
   # Development: 1K rows
   # Staging: 10K rows
   # Production: full dataset
   ```

2. **Cache Validated Artifacts**
   ```yaml
   - name: Cache Verifily
     uses: actions/cache@v4
     with:
       path: .verifily/cache
       key: verifily-${{ hashFiles('data/**') }}
   ```

3. **Async Validation**
   ```python
   from verifily_sdk import AsyncVerifilyClient
   
   async with AsyncVerifilyClient() as client:
       result = await client.pipeline(config_path="verifily.yaml")
   ```

## Synthetic Dataset Generator

Use the included generator for consistent benchmarking:

```python
from examples.benchmark_demo.generate_large_dataset import generate_dataset

# Generate deterministic dataset
generate_dataset(
    output_path="data/test.jsonl",
    num_rows=50000,
    schema="sft",  # or "qa"
    seed=42,       # for determinism
)
```

Generate full benchmark suite:

```bash
python examples/benchmark_demo/generate_large_dataset.py \
    --output ./benchmark_data \
    --suite
```

This creates:
- `sft_1k.jsonl` (1,000 rows)
- `sft_10k.jsonl` (10,000 rows)
- `sft_50k.jsonl` (50,000 rows)
- `qa_1k.jsonl` (1,000 rows)
- etc.

## Measuring in Production

### Prometheus Metrics

```
verifily_latency_ms{quantile="0.95"} 120.000
verifily_endpoint_latency_ms{path="/v1/pipeline",method="POST",quantile="0.95"} 150.000
```

### Custom Benchmarks

```python
import time
from verifily_sdk import VerifilyClient

client = VerifilyClient()

start = time.monotonic()
result = client.pipeline(config_path="verifily.yaml")
elapsed_ms = (time.monotonic() - start) * 1000

print(f"Pipeline took {elapsed_ms:.1f} ms")
```

## Expected Performance

Based on standard cloud instances (AWS m5.large):

| Operation | 1K rows | 10K rows | 100K rows |
|-----------|---------|----------|-----------|
| Ingest | 50ms | 200ms | 1.5s |
| Schema Validation | 30ms | 150ms | 1s |
| Contamination | 100ms | 500ms | 3s |
| Full Pipeline | 200ms | 1s | 6s |

*These are approximate values. Your results may vary.*

## Troubleshooting Slow Performance

### Diagnose

```bash
# Check if it's I/O bound
verifily benchmark --dataset data/large.jsonl
# If rows/sec < 1000, likely I/O bound

# Check memory usage
verifily benchmark --dataset data/large.jsonl
# If memory_mb > 500, consider streaming mode
```

### Solutions

| Problem | Solution |
|---------|----------|
| Slow I/O | Use SSD, local storage |
| High memory | Enable streaming mode |
| Slow validation | Reduce contracts, use sampling |
| Network latency | Use local Verifily instance |

## See Also

- [Benchmark Demo](../scripts/demo_benchmark.sh)
- [Synthetic Dataset Generator](../examples/benchmark_demo/generate_large_dataset.py)
- [Metrics](./metrics.md)
