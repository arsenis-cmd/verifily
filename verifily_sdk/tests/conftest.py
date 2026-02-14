"""Shared test fixtures for Verifily SDK tests."""

from __future__ import annotations

import pytest

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store


@pytest.fixture(scope="module")
def shared_app():
    """Module-scoped app: one create_app() per test module."""
    app = create_app()
    jobs_store.stop_worker()
    return app
