"""Performance guardrails — structural + opt-in wall-clock.

Ensures test infrastructure uses drain() instead of sleep-based polling.
Wall-clock guard: set VERIFILY_PERF_GUARD=1 to assert suite runtime < threshold.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import pytest

from verifily_cli_v1.core.api.jobs import JobsStore

TESTS_DIR = Path(__file__).resolve().parent
SDK_TESTS_DIR = TESTS_DIR.parent.parent / "verifily_sdk" / "tests"

# Files that test async jobs (should use drain, not polling)
JOB_TEST_FILES = [
    TESTS_DIR / "test_api_jobs.py",
    TESTS_DIR / "test_api_jobs_classify.py",
    TESTS_DIR / "test_classify_exports.py",
    SDK_TESTS_DIR / "test_sdk_jobs.py",
    SDK_TESTS_DIR / "test_sdk_jobs_classify.py",
]


class TestDrainHookExists:
    def test_jobs_store_has_drain(self):
        assert hasattr(JobsStore, "drain")

    def test_drain_is_callable(self):
        store = JobsStore()
        assert callable(store.drain)

    def test_drain_returns_count(self):
        store = JobsStore()
        result = store.drain()
        assert isinstance(result, int)
        assert result == 0  # no jobs queued


class TestNoPollingSleep:
    """Verify that job test files do not use time.sleep-based polling."""

    @pytest.mark.parametrize("test_file", JOB_TEST_FILES, ids=lambda p: p.name)
    def test_no_sleep_in_test_helpers(self, test_file):
        """Test helpers must not call time.sleep (use drain() instead)."""
        if not test_file.exists():
            pytest.skip(f"{test_file.name} not found")

        source = test_file.read_text()
        tree = ast.parse(source)

        # Find all function definitions that look like wait/poll helpers
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and any(
                kw in node.name.lower() for kw in ("_wait", "_poll", "wait_for")
            ):
                # Check if the function body contains time.sleep
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute) and func.attr == "sleep":
                            pytest.fail(
                                f"{test_file.name}:{node.lineno} "
                                f"function '{node.name}' uses time.sleep — "
                                f"use jobs_store.drain() instead"
                            )

    @pytest.mark.parametrize("test_file", JOB_TEST_FILES, ids=lambda p: p.name)
    def test_uses_drain_or_no_polling(self, test_file):
        """Job test files should reference drain() somewhere."""
        if not test_file.exists():
            pytest.skip(f"{test_file.name} not found")

        source = test_file.read_text()
        # Either uses drain() or doesn't need it (submit-only tests)
        has_drain = "drain()" in source
        has_submit = "submit" in source.lower() and ("result" in source.lower() or "SUCCEEDED" in source)

        if has_submit:
            assert has_drain, (
                f"{test_file.name} submits jobs and checks results "
                f"but does not use jobs_store.drain()"
            )


class TestModuleScopedFixtures:
    """Verify that conftest provides shared_app for test performance."""

    def test_cli_conftest_exists(self):
        conftest = TESTS_DIR / "conftest.py"
        assert conftest.exists(), "verifily_cli_v1/tests/conftest.py missing"
        source = conftest.read_text()
        assert "shared_app" in source

    def test_sdk_conftest_exists(self):
        conftest = SDK_TESTS_DIR / "conftest.py"
        assert conftest.exists(), "verifily_sdk/tests/conftest.py missing"
        source = conftest.read_text()
        assert "shared_app" in source


# ── Wall-clock guard (opt-in) ────────────────────────────────────

PERF_THRESHOLD_SECONDS = 3.5  # generous ceiling; includes ~0.5s subprocess overhead; target suite time ≤ 2.2s


@pytest.mark.skipif(
    os.environ.get("VERIFILY_PERF_GUARD") != "1",
    reason="Set VERIFILY_PERF_GUARD=1 to enable wall-clock performance guard.",
)
def test_full_suite_under_threshold():
    """Assert full test suite runs under PERF_THRESHOLD_SECONDS.

    Runs the suite in a subprocess (excluding this file) and checks wall time.
    """
    import subprocess
    import sys
    import time

    start = time.monotonic()
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "verifily_cli_v1/tests/", "verifily_sdk/tests/",
            "-q",
            "--ignore=verifily_cli_v1/tests/test_perf_guard.py",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    elapsed = time.monotonic() - start

    last_line = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
    print(f"\n  Suite result: {last_line}")
    print(f"  Wall time:    {elapsed:.2f}s (threshold: {PERF_THRESHOLD_SECONDS}s)")

    assert result.returncode == 0, f"Suite failed:\n{result.stdout}\n{result.stderr}"
    assert elapsed < PERF_THRESHOLD_SECONDS, (
        f"Suite too slow: {elapsed:.2f}s > {PERF_THRESHOLD_SECONDS}s.\n"
        f"Run with --durations=10 to find bottlenecks."
    )
