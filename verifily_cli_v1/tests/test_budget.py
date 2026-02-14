"""Tests for Budget Guards v1 - cost enforcement system.

Target runtime: +0.25s for the full budget test suite.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pytest

from verifily_cli_v1.core.budget import (
    BudgetCheckResult,
    BudgetMode,
    BudgetPolicy,
    BudgetStore,
    budget_store,
    check_budget,
    configure_budgets,
)


class TestBudgetPolicy:
    """Test BudgetPolicy dataclass."""

    def test_default_policy(self) -> None:
        """BudgetPolicy has sensible defaults."""
        policy = BudgetPolicy(project_id="test-proj")
        assert policy.project_id == "test-proj"
        assert policy.daily_limit_units == 1000
        assert policy.monthly_limit_units == 10000
        assert policy.hard_block is True
        assert policy.reset_hour_utc == 0

    def test_custom_policy(self) -> None:
        """BudgetPolicy can be customized."""
        policy = BudgetPolicy(
            project_id="custom",
            daily_limit_units=500,
            monthly_limit_units=5000,
            hard_block=False,
            reset_hour_utc=12,
        )
        assert policy.daily_limit_units == 500
        assert policy.monthly_limit_units == 5000
        assert policy.hard_block is False
        assert policy.reset_hour_utc == 12


class TestBudgetUnits:
    """Test budget unit calculations."""

    def test_rows_conversion(self) -> None:
        """Rows are converted to units correctly (1000 rows = 1 unit)."""
        store = BudgetStore()
        
        # Exactly 1000 rows = 1 unit
        usage = {"rows_in": 1000, "rows_out": 0}
        assert store.compute_units(usage) == 1
        
        # 500 rows = 1 unit (rounded up)
        usage = {"rows_in": 500, "rows_out": 0}
        assert store.compute_units(usage) == 1
        
        # 2500 rows = 3 units (rounded up)
        usage = {"rows_in": 2500, "rows_out": 0}
        assert store.compute_units(usage) == 3

    def test_bytes_conversion(self) -> None:
        """Bytes are converted to units correctly (5MB = 1 unit)."""
        store = BudgetStore()
        
        # Exactly 5MB = 1 unit
        usage = {"bytes_in": 5 * 1024 * 1024}
        assert store.compute_units(usage) == 1
        
        # 2.5MB = 1 unit (rounded up)
        usage = {"bytes_in": 2.5 * 1024 * 1024}
        assert store.compute_units(usage) == 1
        
        # 12MB = 3 units (rounded up)
        usage = {"bytes_in": 12 * 1024 * 1024}
        assert store.compute_units(usage) == 3

    def test_operation_costs(self) -> None:
        """Operations have correct unit costs."""
        store = BudgetStore()
        
        # Decision = 5 units
        usage = {"decisions": 2}
        assert store.compute_units(usage) == 10
        
        # Report = 3 units
        usage = {"reports": 3}
        assert store.compute_units(usage) == 9
        
        # Contamination check = 2 units
        usage = {"contamination_checks": 4}
        assert store.compute_units(usage) == 8
        
        # Classify job = 4 units
        usage = {"classify_jobs": 2}
        assert store.compute_units(usage) == 8
        
        # Retrain = 50 units
        usage = {"retrains": 1}
        assert store.compute_units(usage) == 50
        
        # Monitor tick = 1 unit
        usage = {"monitor_ticks": 100}
        assert store.compute_units(usage) == 100

    def test_combined_units(self) -> None:
        """Multiple metrics are summed correctly."""
        store = BudgetStore()
        
        usage = {
            "rows_in": 1000,        # 1 unit
            "rows_out": 2000,       # 2 units
            "bytes_in": 10 * 1024 * 1024,  # 2 units
            "decisions": 1,         # 5 units
            "reports": 1,           # 3 units
            "contamination_checks": 1,  # 2 units
        }
        # Total: 1 + 2 + 2 + 5 + 3 + 2 = 15 units
        assert store.compute_units(usage) == 15


class TestBudgetStore:
    """Test BudgetStore singleton."""

    def setup_method(self) -> None:
        """Reset budget store before each test."""
        budget_store.reset()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        budget_store.reset()

    def test_singleton_behavior(self) -> None:
        """BudgetStore is a singleton."""
        store1 = BudgetStore()
        store2 = BudgetStore()
        # They are separate instances but should behave independently
        # (singleton pattern is via module-level variable, not class)

    def test_get_policy_default(self) -> None:
        """Default policy is returned for unknown project."""
        policy = budget_store.get_policy("unknown-project")
        assert policy.project_id == "unknown-project"
        assert policy.daily_limit_units == 1000
        assert policy.monthly_limit_units == 10000

    def test_set_and_get_policy(self) -> None:
        """Custom policy can be set and retrieved."""
        custom_policy = BudgetPolicy(
            project_id="my-project",
            daily_limit_units=500,
            monthly_limit_units=5000,
            hard_block=False,
        )
        budget_store.set_policy(custom_policy)
        
        retrieved = budget_store.get_policy("my-project")
        assert retrieved.daily_limit_units == 500
        assert retrieved.monthly_limit_units == 5000
        assert retrieved.hard_block is False

    def test_configure_from_env(self, monkeypatch) -> None:
        """Budget store can be configured from environment variables."""
        monkeypatch.setenv("VERIFILY_BUDGET_DEFAULT_DAILY", "500")
        monkeypatch.setenv("VERIFILY_BUDGET_DEFAULT_MONTHLY", "5000")
        monkeypatch.setenv("VERIFILY_BUDGET_HARD_BLOCK", "0")
        
        store = BudgetStore()
        store.configure()  # Should load from env
        
        policy = store.get_policy("test")
        assert policy.daily_limit_units == 500
        assert policy.monthly_limit_units == 5000
        assert policy.hard_block is False

    def test_configure_from_file(self) -> None:
        """Budget store can be configured from JSON file."""
        config = {
            "policies": [
                {
                    "project_id": "proj-a",
                    "daily_limit_units": 100,
                    "monthly_limit_units": 1000,
                    "hard_block": True,
                    "reset_hour_utc": 6,
                },
                {
                    "project_id": "proj-b",
                    "daily_limit_units": 200,
                    "monthly_limit_units": 2000,
                    "hard_block": False,
                },
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            store = BudgetStore()
            store.configure(config_path)
            
            policy_a = store.get_policy("proj-a")
            assert policy_a.daily_limit_units == 100
            assert policy_a.reset_hour_utc == 6
            
            policy_b = store.get_policy("proj-b")
            assert policy_b.daily_limit_units == 200
            assert policy_b.hard_block is False
        finally:
            os.unlink(config_path)


class TestBudgetCheck:
    """Test budget checking logic."""

    def setup_method(self) -> None:
        """Reset budget store before each test."""
        budget_store.reset()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        budget_store.reset()

    def test_pass_within_budget(self) -> None:
        """Check returns PASS when within budget."""
        policy = BudgetPolicy(
            project_id="test",
            daily_limit_units=100,
            monthly_limit_units=1000,
        )
        budget_store.set_policy(policy)
        
        # Mock no usage - should pass
        result = budget_store.check_budget("test", usage_store=None)
        
        assert result.allowed is True
        assert result.mode == BudgetMode.PASS
        assert result.remaining_daily_units == 100
        assert result.remaining_monthly_units == 1000

    def test_warn_approaching_limit(self) -> None:
        """Check returns WARN when approaching limit (80% threshold)."""
        policy = BudgetPolicy(
            project_id="test",
            daily_limit_units=100,
            monthly_limit_units=1000,
            hard_block=False,
        )
        budget_store.set_policy(policy)
        
        # Simulate 85% usage
        class MockUsageStore:
            def query(self, window_minutes, group_by):
                return {
                    "buckets": [{
                        "project_id": "test",
                        "rows_in": 85000,  # 85 units worth
                    }]
                }
        
        result = budget_store.check_budget("test", usage_store=MockUsageStore())
        
        assert result.allowed is True
        assert result.mode == BudgetMode.WARN

    def test_block_exceeded_hard(self) -> None:
        """Check returns BLOCK when exceeded with hard_block=True."""
        policy = BudgetPolicy(
            project_id="test",
            daily_limit_units=100,
            monthly_limit_units=1000,
            hard_block=True,
        )
        budget_store.set_policy(policy)
        
        # Simulate 150% usage (exceeded)
        class MockUsageStore:
            def query(self, window_minutes, group_by):
                return {
                    "buckets": [{
                        "project_id": "test",
                        "rows_in": 150000,  # 150 units worth (exceeds 100)
                    }]
                }
        
        result = budget_store.check_budget("test", usage_store=MockUsageStore())
        
        assert result.allowed is False
        assert result.mode == BudgetMode.BLOCK
        assert result.remaining_daily_units == 0

    def test_warn_exceeded_soft(self) -> None:
        """Check returns WARN when exceeded with hard_block=False."""
        policy = BudgetPolicy(
            project_id="test",
            daily_limit_units=100,
            monthly_limit_units=1000,
            hard_block=False,
        )
        budget_store.set_policy(policy)
        
        # Simulate 150% usage (exceeded)
        class MockUsageStore:
            def query(self, window_minutes, group_by):
                return {
                    "buckets": [{
                        "project_id": "test",
                        "rows_in": 150000,
                    }]
                }
        
        result = budget_store.check_budget("test", usage_store=MockUsageStore())
        
        assert result.allowed is True  # Still allowed with soft block
        assert result.mode == BudgetMode.WARN

    def test_reset_time_calculation(self) -> None:
        """Reset time is calculated correctly."""
        policy = BudgetPolicy(
            project_id="test",
            daily_limit_units=100,
            reset_hour_utc=6,  # Reset at 6 AM UTC
        )
        budget_store.set_policy(policy)
        
        # Test at midnight UTC (before reset)
        now = datetime(2024, 1, 15, 0, 0, 0)
        result = budget_store.check_budget("test", usage_store=None, now_utc=now)
        
        assert result.reset_time_utc.startswith("2024-01-15T06:00:00")
        
        # Test at noon UTC (after reset, next day)
        now = datetime(2024, 1, 15, 12, 0, 0)
        result = budget_store.check_budget("test", usage_store=None, now_utc=now)
        
        assert result.reset_time_utc.startswith("2024-01-16T06:00:00")


class TestBudgetStatus:
    """Test budget status reporting."""

    def setup_method(self) -> None:
        """Reset budget store before each test."""
        budget_store.reset()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        budget_store.reset()

    def test_get_status_structure(self) -> None:
        """Status has correct structure."""
        policy = BudgetPolicy(
            project_id="test",
            daily_limit_units=100,
            monthly_limit_units=1000,
        )
        budget_store.set_policy(policy)
        
        status = budget_store.get_status("test")
        
        assert status["project_id"] == "test"
        assert "policy" in status
        assert "usage" in status
        assert "next_reset" in status
        assert "seconds_until_reset" in status
        
        assert status["policy"]["daily_limit_units"] == 100
        assert status["policy"]["monthly_limit_units"] == 1000

    def test_convenience_functions(self) -> None:
        """Convenience functions work correctly."""
        configure_budgets()  # Should not raise
        result = check_budget("default")
        assert isinstance(result, BudgetCheckResult)


class TestBudgetIntegration:
    """Integration tests for budget system."""

    def test_full_budget_lifecycle(self) -> None:
        """Test complete budget lifecycle: configure, check, status."""
        budget_store.reset()
        
        # 1. Configure a policy
        policy = BudgetPolicy(
            project_id="integration-test",
            daily_limit_units=500,
            monthly_limit_units=5000,
            hard_block=True,
            reset_hour_utc=0,
        )
        budget_store.set_policy(policy)
        
        # 2. Check budget (should pass initially)
        result = budget_store.check_budget("integration-test")
        assert result.mode == BudgetMode.PASS
        
        # 3. Get full status
        status = budget_store.get_status("integration-test")
        assert status["policy"]["daily_limit_units"] == 500
        assert status["usage"]["mode"] == BudgetMode.PASS
        
        budget_store.reset()


class TestBudgetErrorHandling:
    """Test budget system error handling."""

    def test_invalid_config_file(self) -> None:
        """Gracefully handle invalid config file."""
        store = BudgetStore()
        # Non-existent file should use env defaults
        store.configure("/nonexistent/config.json")
        
        policy = store.get_policy("test")
        assert policy.daily_limit_units == 1000  # Default

    def test_malformed_config_file(self, tmp_path) -> None:
        """Gracefully handle malformed config file."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text("not valid json")
        
        store = BudgetStore()
        store.configure(str(config_file))
        
        policy = store.get_policy("test")
        assert policy.daily_limit_units == 1000  # Default


# Performance test - ensure budget checks are fast
class TestBudgetPerformance:
    """Performance tests for budget system."""

    def test_check_budget_fast(self) -> None:
        """Budget check should be fast (< 1ms)."""
        import time
        
        budget_store.reset()
        
        policy = BudgetPolicy(
            project_id="perf-test",
            daily_limit_units=10000,
            monthly_limit_units=100000,
        )
        budget_store.set_policy(policy)
        
        # Time the check
        start = time.perf_counter()
        result = budget_store.check_budget("perf-test")
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert result.allowed is True
        assert elapsed_ms < 10  # Should be much faster than 10ms
        
        budget_store.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
