"""Repo-wide test fixtures.

Snapshots and restores sensitive environment variables between tests
to prevent cross-module pollution from module-level os.environ mutations.
"""

from __future__ import annotations

import os

import pytest

_SENSITIVE_ENV_VARS = [
    "VERIFILY_ORG_MODE",
    "VERIFILY_API_KEY",
    "VERIFILY_ALLOW_NONLOCAL",
    "VERIFILY_PROD",
    "VERIFILY_WORKSPACE_ROOT",
    "VERIFILY_ALLOW_ABSPATH",
    "VERIFILY_RATE_LIMIT_RPM",
    "VERIFILY_USAGE_PERSIST",
    "VERIFILY_JOBS_PERSIST",
    "VERIFILY_BILLING_ENABLED",
    "VERIFILY_BILLING_STRIPE_KEY",
    "VERIFILY_BILLING_WEBHOOK_SECRET",
    "VERIFILY_SKIP_SIGNALS",
    "VERIFILY_WORKSPACES_ENABLED",
    "VERIFILY_BOOTSTRAP_TOKEN",
    "VERIFILY_WORKSPACES_STORE_PATH",
    "VERIFILY_KEY_SALT",
    "VERIFILY_TEAMS_ENABLED",
    "VERIFILY_ENTERPRISE_TOKEN",
]


@pytest.fixture(autouse=True)
def _restore_env():
    """Snapshot sensitive env vars before each test and restore after."""
    snapshot = {}
    for var in _SENSITIVE_ENV_VARS:
        val = os.environ.get(var)
        if val is not None:
            snapshot[var] = val

    yield

    # Restore: remove any that were added, reset any that changed
    for var in _SENSITIVE_ENV_VARS:
        if var in snapshot:
            os.environ[var] = snapshot[var]
        else:
            os.environ.pop(var, None)
