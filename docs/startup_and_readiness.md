# Startup Checks and Readiness

This document describes Verifily's startup validation, readiness probes, and graceful shutdown behavior.

## Overview

Verifily performs comprehensive environment validation before accepting any connections. This ensures production deployments fail fast with clear error messages rather than failing silently during operation.

## Startup Checks

### What Gets Validated

When the server starts, the following checks are performed:

| Check | Fatal | Description |
|-------|-------|-------------|
| Python version | Yes | Requires Python >= 3.9 |
| Data directory writable | Yes | Must have write access to `--data-dir` |
| Usage persistence writable | Yes | If `VERIFILY_USAGE_PERSIST=1`, path must be writable |
| Jobs persistence writable | Yes | If `VERIFILY_JOBS_PERSIST=1`, path must be writable |
| Temp directory writable | Yes | System temp dir must be writable |
| API key format | Yes | If `VERIFILY_API_KEY` is set, it must be non-empty |
| Rate limit config | Yes | `VERIFILY_RATE_LIMIT_RPM` must be >= 0 if set |
| Production auth | No | Warns if `VERIFILY_ENV=prod` without `VERIFILY_API_KEY` |

### Check Results

```python
class StartupCheckResult(BaseModel):
    ok: bool           # True if no fatal errors
    warnings: List[str]  # Non-fatal issues (logged but continue)
    errors: List[str]    # Fatal issues (prevent startup)
```

### Behavior

**Errors** → Server refuses to start with a clear error message:

```
Verifily Startup Checks
  ✓ Python version OK
  ✗ No write permission for data directory: /data
  ✗ VERIFILY_API_KEY is set but empty

Startup failed: Environment validation errors detected.
Fix the issues above and try again.
```

**Warnings** → Server starts but logs warnings:

```
Verifily Startup Checks
  ✓ Python version OK
  ✓ Data directory writable
  ⚠ Rate limit not configured (unlimited)
  ⚠ Running in prod without API key authentication

✓ Startup checks passed
```

## Readiness Endpoint

The `/ready` endpoint returns the current readiness state of the server.

### Endpoint

```
GET /ready
```

### Ready State (HTTP 200)

```json
{
  "status": "ready",
  "checks": {
    "startup": "ok",
    "python": "ok",
    "temp_write": "ok",
    "imports": "ok",
    "jobs_store": "ok",
    "usage_store": "ok",
    "monitor_store": "ok",
    "jobs_worker": "alive",
    "usage_persistence": "configured"
  }
}
```

### Not Ready State (HTTP 503)

```json
{
  "status": "not_ready",
  "checks": {
    "startup": "failed"
  },
  "error": "Startup checks failed: No write permission for data directory: /data"
}
```

### Readiness Criteria

The server reports **ready** only when:

1. ✅ Startup checks passed
2. ✅ Python version >= 3.9
3. ✅ Temp directory writable
4. ✅ Critical imports loadable
5. ✅ Subsystem stores initialized
6. ✅ JobsStore worker thread alive
7. ✅ Persistence initialized (if enabled)
8. ✅ Server not shutting down

## Graceful Shutdown

When the server receives a `SIGTERM` or `SIGINT`:

```
Verifily shutting down gracefully...
  ✓ JobsStore stopped
  ✓ MonitorStore stopped
  ✓ UsageStore healthy
Graceful shutdown complete.
```

### Shutdown Process

1. **Stop accepting new connections** (uvicorn)
2. **Stop JobsStore worker thread** (wait up to 5s for current job)
3. **Stop all MonitorStore threads** (signal stop, wait for threads)
4. **Verify persistence health** (test write to ensure buffers flushed)
5. **Exit**

No dangling threads remain after shutdown.

## Docker Health Checks

The Dockerfile includes a `HEALTHCHECK` instruction:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/health >/dev/null || exit 1
```

This uses `/health` (not `/ready`) for liveness because:

- `/health` = "Is the process alive?" (always returns 200 if process runs)
- `/ready` = "Is the server ready to serve traffic?" (503 during startup/shutdown)

For Kubernetes, use:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  periodSeconds: 10
```

## Production Environment Variables

### Required

None. Verifily runs with sensible defaults.

### Recommended for Production

| Variable | Description | Example |
|----------|-------------|---------|
| `VERIFILY_ENV` | Set to `prod` to enable production warnings | `prod` |
| `VERIFILY_API_KEY` | Enable API key authentication | `sk-...` |
| `VERIFILY_DATA_DIR` | Persistent data directory | `/data` |
| `VERIFILY_RATE_LIMIT_RPM` | Rate limit per minute per key | `100` |

### Optional Persistence

| Variable | Description | Default |
|----------|-------------|---------|
| `VERIFILY_USAGE_PERSIST` | Enable usage accounting persistence | `0` |
| `VERIFILY_USAGE_LOG_PATH` | Path to usage events JSONL | `{data_dir}/verifily_usage_events.jsonl` |
| `VERIFILY_JOBS_PERSIST` | Enable job history persistence | `0` |
| `VERIFILY_JOBS_LOG_PATH` | Path to jobs events JSONL | `{data_dir}/verifily_jobs_events.jsonl` |

## Troubleshooting

### "No write permission for data directory"

The data directory must be writable by the verifily user:

```bash
# Fix permissions in Docker
RUN mkdir -p /data && chown -R verifily:verifily /data

# Or mount with correct permissions
docker run -v $(pwd)/data:/data verifily
```

### "Startup checks not completed"

The `/ready` endpoint returns this during the brief window before startup checks complete. This should resolve within milliseconds.

### "JobsStore worker thread is not running"

The worker thread failed to start or crashed. Check logs for earlier errors. The server will report not-ready until the worker is healthy.

### Graceful shutdown timeout

If shutdown takes longer than expected, the server forcefully terminates threads after:

- JobsStore: 5 seconds
- MonitorStore: 2 seconds per monitor

Check logs for:
- Long-running jobs blocking shutdown
- Monitor threads stuck on I/O
- Persistence write failures

## Verification Commands

### Test startup checks locally

```bash
# With invalid config (should fail)
VERIFILY_API_KEY="" verifily serve

# With valid config
verifily serve --port 8080
```

### Test readiness probe

```bash
# Should return 503 during startup, then 200
curl -f http://localhost:8080/ready || echo "Not ready"
```

### Test graceful shutdown

```bash
# Start server
verifily serve --port 8080 &
PID=$!

# Trigger shutdown
kill -TERM $PID

# Watch logs for graceful shutdown messages
```

### Docker health check

```bash
docker compose up -d
docker compose ps  # Should show "healthy"
docker compose exec verifily curl -f http://localhost:8080/ready
```
