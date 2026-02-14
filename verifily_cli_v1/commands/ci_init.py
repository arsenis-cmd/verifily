"""verifily ci init — generate CI pipeline templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from verifily_cli_v1.core.io import ensure_dir

# ── Templates ──────────────────────────────────────────────────────

GITHUB_TEMPLATE = """\
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
"""

GITLAB_TEMPLATE = """\
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
"""


# ── Public API ─────────────────────────────────────────────────────

def generate_ci(
    provider: str,
    project_root: str = ".",
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Generate a CI config file for the given provider.

    Args:
        provider: "github" or "gitlab".
        project_root: Project root directory.
        force: Overwrite existing file.

    Returns:
        Dict with provider, file_path, project_root.
    """
    root = Path(project_root).resolve()

    if provider == "github":
        rel_path = ".github/workflows/verifily.yml"
        content = GITHUB_TEMPLATE
    elif provider == "gitlab":
        rel_path = ".gitlab-ci.yml"
        content = GITLAB_TEMPLATE
    else:
        raise ValueError(f"Unknown CI provider: {provider!r}. Use 'github' or 'gitlab'.")

    file_path = root / rel_path

    if file_path.exists() and not force:
        raise FileExistsError(
            f"CI config already exists: {file_path}\n"
            "Use --force to overwrite."
        )

    ensure_dir(file_path.parent)
    file_path.write_text(content, encoding="utf-8")

    return {
        "provider": provider,
        "file_path": str(file_path),
        "relative_path": rel_path,
        "project_root": str(root),
    }
