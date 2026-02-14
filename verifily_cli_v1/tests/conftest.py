"""Shared test fixtures for Verifily CLI tests.

Provides a module-scoped app fixture to avoid re-creating FastAPI apps
for every single test (create_app takes ~10ms × 187 tests = ~2s overhead).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.monitor_store import monitor_store


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"


@pytest.fixture(scope="module")
def shared_app(request):
    """Module-scoped app: one create_app() per test module."""
    app = create_app()
    jobs_store.stop_worker()
    return app
