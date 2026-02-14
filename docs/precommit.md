# Pre-commit Hook

Verifily provides a pre-commit hook that blocks commits when the pipeline gate returns DONT_SHIP or CONTRACT_FAIL.

## Behavior

| Exit Code | Meaning | Commit Allowed |
|-----------|---------|----------------|
| 0 | SHIP | Yes |
| 1 | DONT_SHIP | No — commit blocked |
| 2 | INVESTIGATE | Yes — investigation doesn't block |
| 3 | CONTRACT_FAIL | No — commit blocked |
| 4 | TOOL_ERROR | Yes — don't block on tool issues |

If no `verifily.yaml` is found, the hook exits silently (no block).

## Setup: pre-commit framework

Add to your `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: verifily-gate
        name: Verifily pipeline gate
        entry: bash scripts/precommit_verifily.sh
        language: system
        always_run: true
        pass_filenames: false
```

## Setup: Manual git hook

Copy the script directly:

```bash
cp scripts/precommit_verifily.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Configuration

Set `VERIFILY_CONFIG` to override the config path:

```bash
export VERIFILY_CONFIG=configs/verifily.yaml
```

Default: `verifily.yaml` in the current directory.

## When to Enable

Enable the pre-commit hook when:
- Your project has a stable `verifily.yaml` and baseline run
- You want to catch contamination or contract violations before push
- Your pipeline runs fast enough for commit-time checks

Disable or skip when:
- Initial project setup (no baseline yet)
- Large datasets that make pipeline slow
- Use `git commit --no-verify` to skip temporarily
