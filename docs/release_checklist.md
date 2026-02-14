# Release Checklist

Steps for cutting a Verifily release.

## Pre-release

### 1. Version bump

Update the version string in all four locations:

```bash
# 1. Root VERSION file
echo "1.x.y" > VERSION

# 2. CLI __init__.py
# verifily_cli_v1/__init__.py → __version__ = "1.x.y"

# 3. Root pyproject.toml
# [project] version = "1.x.y"

# 4. SDK
# verifily_sdk/verifily_sdk/__init__.py → __version__ = "1.x.y"
# verifily_sdk/pyproject.toml → version = "1.x.y"
```

Verify alignment:

```bash
python3 -c "
from verifily_cli_v1 import __version__ as cli_v
from verifily_sdk import __version__ as sdk_v
v = open('VERSION').read().strip()
assert cli_v == sdk_v == v, f'Mismatch: CLI={cli_v}, SDK={sdk_v}, VERSION={v}'
print(f'All versions aligned: {v}')
"
```

### 2. Run full test suite with perf guard

```bash
python3 -m pytest verifily_cli_v1/tests/ verifily_sdk/tests/ -q
```

Expected: all tests pass, runtime under 3s.

### 3. Run demo scripts

```bash
bash scripts/demo_quickstart.sh
bash scripts/demo_quickstart_ci.sh
bash scripts/demo_real_conditions.sh
bash scripts/demo_fingerprint.sh
```

All must exit 0.

### 4. Doctor check

```bash
verifily doctor
```

Expected: 0 (healthy) or 2 (warnings only, no failures).

### 5. Build Docker image

```bash
docker compose build
docker compose up -d
curl -s http://localhost:8080/health | python3 -m json.tool
docker compose down
```

### 6. Smoke install in fresh venv

```bash
python3 -m venv /tmp/verifily_smoke
source /tmp/verifily_smoke/bin/activate
pip install -e ".[all]"
pip install -e verifily_sdk/
verifily --version
verifily doctor
verifily quickstart /tmp/smoke_project
bash /tmp/smoke_project/scripts/run_demo.sh
deactivate
rm -rf /tmp/verifily_smoke /tmp/smoke_project
```

## Release

### 7. Update CHANGELOG.md

Add a section for the new version with:
- Added (new features)
- Changed (modifications)
- Fixed (bug fixes)
- Removed (if any)

### 8. Tag

```bash
git tag -a v1.x.y -m "Release v1.x.y"
git push origin v1.x.y
```

## Post-release

### 9. Verify Docker image tag

```bash
docker compose build
docker compose up -d
curl -s http://localhost:8080/health | python3 -c "import json,sys; print(json.load(sys.stdin)['version'])"
docker compose down
```

Version in `/health` should match the tag.
