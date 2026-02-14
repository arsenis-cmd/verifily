# CI Init — Quick CI Setup

Generate copy-paste CI config files for GitHub Actions or GitLab CI.

## Usage

```bash
# GitHub Actions
verifily ci-init --github

# GitLab CI
verifily ci-init --gitlab

# Custom project root
verifily ci-init --github --path ./my_project
```

## What it creates

### GitHub Actions

File: `.github/workflows/verifily.yml`

```yaml
name: Verifily Gate
on: [push, pull_request]

jobs:
  verifily-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install -e .
      - name: Run Verifily Gate
        run: |
          set -o pipefail
          verifily pipeline --ci --json 2>&1 | tee verifily_result.json
      - name: Upload result on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: verifily-result
          path: verifily_result.json
```

### GitLab CI

File: `.gitlab-ci.yml`

```yaml
verifily-gate:
  stage: test
  image: python:3.10
  script:
    - pip install -e .
    - set -o pipefail
    - verifily pipeline --ci --json 2>&1 | tee verifily_result.json
  artifacts:
    when: always
    paths:
      - verifily_result.json
    expire_in: 30 days
```

## Customization

### API key (optional)

If using the Verifily API server, add an environment variable:

```yaml
# GitHub: add to your workflow
env:
  VERIFILY_API_KEY: ${{ secrets.VERIFILY_API_KEY }}

# GitLab: add to your CI variables
variables:
  VERIFILY_API_KEY: $VERIFILY_API_KEY
```

### Server URL (optional)

```yaml
env:
  VERIFILY_SERVER: https://your-server.example.com
```

## Options

```bash
verifily ci-init --github|--gitlab [--path <dir>] [--force] [--json]
```

| Flag | Description |
|------|-------------|
| `--github` | Generate GitHub Actions workflow |
| `--gitlab` | Generate GitLab CI config |
| `--path` | Project root (default: `.`) |
| `--force` | Overwrite existing CI files |
| `--json` | Output JSON with file paths |
