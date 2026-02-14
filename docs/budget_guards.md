# Budget Guards v1

Cost enforcement system to prevent unexpected runaway spending in Verifily.

## Overview

Budget Guards provide per-project cost controls with daily and monthly limits. When limits are exceeded, requests can either be blocked (hard enforcement) or warned (soft enforcement).

## Quick Start

### CLI

```bash
# Check budget status
verifily budget status
verifily budget status --project my-project

# Set custom policy
verifily budget set-policy --project my-project --daily 1000 --monthly 10000

# Soft enforcement (warn only)
verifily budget set-policy --project my-project --daily 1000 --monthly 10000 --soft-block
```

### SDK

```python
from verifily_sdk import VerifilyClient

client = VerifilyClient()

# Check budget status
status = client.budget_status(project_id="my-project")
print(f"Mode: {status.usage.mode}")
print(f"Daily: {status.usage.daily_used}/{status.usage.daily_limit}")
print(f"Monthly: {status.usage.monthly_used}/{status.usage.monthly_limit}")
print(f"Next reset: {status.next_reset}")
```

### API

```bash
# Get budget status
curl "http://localhost:8000/v1/budget/status?project_id=my-project"

# Response includes budget headers on all billable endpoints
curl -X POST "http://localhost:8000/v1/pipeline" \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/path/to/config.yaml"}'

# Response headers:
# X-Budget-Mode: pass
# X-Budget-Remaining-Daily: 985
# X-Budget-Remaining-Monthly: 9845
```

## Budget Units

Budget consumption is measured in abstract "units" calculated as:

```
units = ceil(rows_in / 1000)
      + ceil(rows_out / 1000)
      + ceil(bytes_in / 5MB)
      + ceil(bytes_out / 5MB)
      + decisions * 5
      + reports * 3
      + contamination_checks * 2
      + classify_jobs * 4
      + retrains * 50
      + monitor_ticks * 1
```

### Unit Costs

| Operation | Unit Cost |
|-----------|-----------|
| 1,000 rows processed | 1 unit |
| 5 MB data processed | 1 unit |
| Pipeline decision | 5 units |
| Report generated | 3 units |
| Contamination check | 2 units |
| Classify job | 4 units |
| Retrain job | 50 units |
| Monitor tick | 1 unit |

## Configuration

### Environment Variables

```bash
# Default policy (applied to all projects without specific policy)
export VERIFILY_BUDGET_DEFAULT_DAILY=1000
export VERIFILY_BUDGET_DEFAULT_MONTHLY=10000
export VERIFILY_BUDGET_HARD_BLOCK=1
export VERIFILY_BUDGET_RESET_HOUR=0  # UTC hour for daily reset
```

### JSON Configuration File

Create `budget_config.json`:

```json
{
  "policies": [
    {
      "project_id": "prod-project",
      "daily_limit_units": 5000,
      "monthly_limit_units": 50000,
      "hard_block": true,
      "reset_hour_utc": 0
    },
    {
      "project_id": "dev-project",
      "daily_limit_units": 100,
      "monthly_limit_units": 1000,
      "hard_block": false,
      "reset_hour_utc": 6
    }
  ]
}
```

Load it:

```python
from verifily_cli_v1.core.budget import configure_budgets

configure_budgets("/path/to/budget_config.json")
```

## Enforcement Modes

### PASS (Within Budget)
- All requests proceed normally
- Budget headers included in responses
- Mode: `pass`

### WARN (Approaching Limit)
- Requests proceed
- Warning headers included
- Triggered at 80% of any limit
- Mode: `warn`

### BLOCK (Limit Exceeded)
- Requests rejected with HTTP 402 (Payment Required)
- `Retry-After` header indicates when budget resets
- Only with `hard_block: true`
- Mode: `block`

## Error Responses

When budget is exceeded (hard block):

```http
HTTP/1.1 402 Payment Required
Retry-After: 3600
X-Budget-Remaining-Daily: 0
X-Budget-Remaining-Monthly: 845
X-Budget-Reset-Time: 2024-01-16T00:00:00Z

{
  "error": {
    "type": "BUDGET_EXCEEDED",
    "message": "Budget exceeded: daily=1000/1000, monthly=5155/10000",
    "code": "budget_limit_reached"
  }
}
```

## Billable Endpoints

Budget checks apply to:

- `POST /v1/pipeline`
- `POST /v1/report`
- `POST /v1/contamination`
- `POST /v1/jobs/*`
- `PUT /v1/jobs/*`
- `DELETE /v1/jobs/*`
- `POST /v1/monitor/start`

Non-billable endpoints (always allowed):

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/budget/*`

## Best Practices

1. **Set Conservative Limits**: Start with lower limits and increase as needed
2. **Use Soft Block in Dev**: Set `hard_block: false` for development projects
3. **Monitor Regularly**: Check budget status in CI/CD pipelines
4. **Set Reset Hour**: Align with your billing cycle or off-peak hours
5. **Per-Project Policies**: Different limits for prod vs dev projects

## Integration with Jobs

The JobsStore integrates with budget checking:

```python
from verifily_cli_v1.core.api.jobs import JobType, jobs_store, BudgetExceededError

try:
    job_id = jobs_store.submit(
        JobType.PIPELINE,
        payload={"config_path": "/path/to/config.yaml"},
        project_id="my-project",
    )
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
    print(f"Reset at: {e.budget_result.reset_time_utc}")
```

## Testing

Run budget tests:

```bash
pytest verifily_cli_v1/tests/test_budget.py -v
```

Test isolation uses `budget_store.reset()` between tests.

## Troubleshooting

### Budget check fails silently
- Check that usage store is properly configured
- Verify project_id is being extracted correctly

### Unexpected blocking
- Check `VERIFILY_BUDGET_HARD_BLOCK` setting
- Review actual usage with `verifily budget status`
- Verify units calculation matches expectations

### Reset time not working
- Verify `reset_hour_utc` is in 0-23 range
- Check server timezone (all times are UTC)

## Migration Guide

### From v0 (no budget controls)
1. Set environment variables for default policy
2. Monitor for `X-Budget-Warning` headers
3. Gradually enable hard blocking

### From quota system
Budget Guards replace the simpler quota system:
- More granular (daily + monthly)
- Per-project policies
- Better visibility via status endpoint

## Future Enhancements

Planned for v2:
- Webhook notifications when approaching limits
- Budget forecasting based on usage trends
- Team-level budget aggregation
- Cost allocation tags
