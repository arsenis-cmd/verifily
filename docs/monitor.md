# Continuous Gating Monitor — User Guide

Run the Verifily gate repeatedly on dataset artifacts, producing rolling history and regression alerts.

## Flow

```
pipeline.yaml
  → monitor start (interval=60s, max_ticks=100)
    → tick 1: CONTRACT → REPORT → CONTAMINATION → DECISION
    → tick 2: CONTRACT → REPORT → CONTAMINATION → DECISION
    → ...
    → history.jsonl (rolling window)
    → regression alerts (metric delta > 2%)
```

## Quick Start

```bash
bash scripts/demo_monitor.sh
```

## API Endpoints

### Start a monitor

```bash
curl -s -X POST http://localhost:8080/v1/monitor/start \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "/path/to/verifily.yaml",
    "interval_seconds": 60,
    "max_ticks": 100,
    "rolling_window": 20
  }'
```

Response:
```json
{"monitor_id": "abc123def456", "status": "running"}
```

### Stop a monitor

```bash
curl -s -X POST "http://localhost:8080/v1/monitor/stop?monitor_id=abc123def456"
```

### Check status

```bash
curl -s "http://localhost:8080/v1/monitor/status?monitor_id=abc123def456"
```

Response:
```json
{
  "monitor_id": "abc123def456",
  "status": "running",
  "tick_count": 5,
  "last_tick": {
    "tick_number": 5,
    "decision": "SHIP",
    "metric_value": 0.72,
    "delta": 0.0,
    "regression_detected": false,
    "contamination_pass": true,
    "contract_pass": true
  },
  "config": {"..."}
}
```

### Get history

```bash
curl -s "http://localhost:8080/v1/monitor/history?monitor_id=abc123def456&last_n=5"
```

## SDK Usage

```python
from verifily_sdk import VerifilyClient

client = VerifilyClient(base_url="http://localhost:8080")

# Start
resp = client.start_monitor(
    config_path="/path/to/verifily.yaml",
    interval_seconds=60,
    max_ticks=100,
)
print(resp.monitor_id)

# Check status
status = client.monitor_status(resp.monitor_id)
print(status.tick_count, status.last_tick)

# Get history
history = client.monitor_history(resp.monitor_id, last_n=10)
for tick in history.ticks:
    print(tick["decision"], tick["metric_value"])

# Stop
client.stop_monitor(resp.monitor_id)
```

## CLI Commands

```bash
# Start a monitor
verifily monitor-start --config verifily.yaml --interval 60 --max-ticks 100

# Check status
verifily monitor-status --monitor-id abc123def456

# View history
verifily monitor-history --monitor-id abc123def456 --last-n 10

# Stop
verifily monitor-stop --monitor-id abc123def456
```

## Regression Detection

Each tick compares the primary metric (F1) to the previous tick. If the delta drops more than 2%, the tick is flagged with `regression_detected: true`.

## History Persistence

Each tick result is appended to `monitor_history.jsonl` alongside the pipeline config file. The in-memory history is capped at `rolling_window` entries (default 20).
