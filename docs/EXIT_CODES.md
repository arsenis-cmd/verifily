# Verifily Exit Codes

Verifily uses standardized exit codes for CI/CD integration and automation.

## Standard Exit Codes

| Exit Code | Meaning | Description |
|-----------|---------|-------------|
| `0` | **SUCCESS** | All checks passed, quality gate approved |
| `1` | **FAIL** | One or more checks failed, quality gate rejected |
| `2` | **WARN** | Checks passed with warnings, review recommended |
| `3` | **ERROR** | Internal error or unexpected failure |
| `130` | **INTERRUPTED** | User interrupted (Ctrl+C) |

## Exit Code Details

### 0 - SUCCESS
- All contracts passed
- No contamination detected
- No regressions detected
- Decision: **PASS**

**CI Behavior**: Pipeline continues

### 1 - FAIL
- One or more contracts failed
- Contamination detected above threshold
- Regression detected
- Budget exceeded
- Decision: **FAIL**

**CI Behavior**: Pipeline fails by default

### 2 - WARN
- Contracts passed with warnings
- Minor issues detected
- Near-threshold contamination
- Decision: **PASS_WITH_WARNINGS**

**CI Behavior**: Pipeline continues (configurable with `--fail-on warning`)

### 3 - ERROR
- Internal server error
- Configuration error
- Network error
- Invalid arguments

**CI Behavior**: Pipeline fails

### 130 - INTERRUPTED
- User pressed Ctrl+C
- Job was terminated
- Timeout reached

**CI Behavior**: Pipeline cancelled

## CI/CD Configuration

### GitHub Actions

```yaml
- name: Run Verifily Gate
  uses: ./.github/actions/verifily-gate
  with:
    config-path: './verifily.yaml'
    fail-on: 'error'  # Options: error, warning, never
```

### GitLab CI

```yaml
verifily-gate:
  script:
    - verifily pipeline --config verifily.yaml --ci
  allow_failure:
    exit_codes:
      - 2  # Allow warnings
```

### Azure DevOps

```yaml
- script: |
    verifily pipeline --config verifily.yaml --ci
    echo "##vso[task.setvariable variable=gate_exit_code]$?"
  continueOnError: true

- script: |
    if [ "$(gate_exit_code)" -eq 1 ]; then
      echo "Quality gate FAILED"
      exit 1
    elif [ "$(gate_exit_code)" -eq 2 ]; then
      echo "Quality gate WARNING"
    fi
```

## Programmatic Usage

```python
import subprocess

result = subprocess.run(["verifily", "pipeline", "--config", "verifily.yaml", "--ci"])

if result.returncode == 0:
    print("✅ Quality gate passed")
elif result.returncode == 1:
    print("❌ Quality gate failed")
elif result.returncode == 2:
    print("⚠️ Quality gate passed with warnings")
else:
    print(f"💥 Error: exit code {result.returncode}")
```

## Shell Scripting

```bash
#!/bin/bash
set -e  # Exit on error

verifily pipeline --config verifily.yaml --ci
EXIT_CODE=$?

case $EXIT_CODE in
    0)
        echo "✅ Gate passed"
        ;;
    1)
        echo "❌ Gate failed"
        exit 1
        ;;
    2)
        echo "⚠️ Gate passed with warnings"
        # Decide whether to continue
        ;;
    *)
        echo "💥 Unexpected error: $EXIT_CODE"
        exit 1
        ;;
esac
```

## See Also

- [CI/CD Integration](./CI_INTEGRATION.md)
- [GitHub Actions](../.github/actions/verifily-gate/action.yml)
- [GitLab CI Example](../examples/ci/gitlab-ci.yml)
