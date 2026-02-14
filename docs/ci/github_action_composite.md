# Verifily Gate — Composite GitHub Action

A repo-local composite action that runs the Verifily pipeline gate in CI.

## Location

```
.github/actions/verifily-gate/action.yml
```

## Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `config` | `verifily.yaml` | Path to pipeline config file |
| `workdir` | `.` | Working directory containing the project |
| `api_key` | (empty) | Optional Verifily API key |
| `strict` | `true` | Fail the workflow on DONT_SHIP |
| `allow_investigate` | `false` | Allow INVESTIGATE without failing |

## Outputs

| Output | Description |
|--------|-------------|
| `exit_code` | Pipeline exit code (0/1/2/3/4) |
| `decision` | Decision string (SHIP/DONT_SHIP/INVESTIGATE) |

## Exit Code Behavior

| Code | Meaning | strict=true | strict=false | allow_investigate=true |
|------|---------|-------------|--------------|----------------------|
| 0 | SHIP | pass | pass | pass |
| 1 | DONT_SHIP | fail | warn | fail |
| 2 | INVESTIGATE | fail | warn | pass (warn) |
| 3 | CONTRACT_FAIL | fail | warn | fail |
| 4 | TOOL_ERROR | fail | warn | fail |

## Minimal Example

```yaml
name: Verifily Gate
on: [push, pull_request]

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: ./.github/actions/verifily-gate
```

## Example with Options

```yaml
      - uses: ./.github/actions/verifily-gate
        with:
          config: configs/verifily.yaml
          workdir: ml_project
          strict: 'true'
          allow_investigate: 'true'
```

## Example: Use Decision in Later Steps

```yaml
      - uses: ./.github/actions/verifily-gate
        id: gate
      - run: echo "Decision was ${{ steps.gate.outputs.decision }}"
      - if: steps.gate.outputs.decision == 'SHIP'
        run: echo "Deploying..."
```

## Artifacts

The action always uploads:
- `verifily_result.json` — full pipeline JSON output
- `verifily_artifacts/` — decision summary, audit log, usage
- `verifily_stderr.txt` — stderr output (Decision: line)

These appear as downloadable artifacts named `verifily-gate-results`.
