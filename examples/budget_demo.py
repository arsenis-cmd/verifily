#!/usr/bin/env python3
"""
Budget Guards Demo - Showcase cost enforcement features.

This demo shows how to:
1. Configure budget policies
2. Check budget status
3. Handle budget exceeded errors
4. Simulate usage and track consumption

Run: python examples/budget_demo.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from verifily_cli_v1.core.budget import (
    BudgetCheckResult,
    BudgetMode,
    BudgetPolicy,
    BudgetStore,
    budget_store,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_json(data: dict) -> None:
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))


def demo_1_basic_policy() -> None:
    """Demo 1: Create and query a budget policy."""
    print_header("Demo 1: Basic Budget Policy")
    
    # Reset store for clean demo
    budget_store.reset()
    
    # Create a policy
    policy = BudgetPolicy(
        project_id="demo-project",
        daily_limit_units=100,
        monthly_limit_units=1000,
        hard_block=True,
        reset_hour_utc=0,
    )
    
    print(f"Created policy:")
    print(f"  Project: {policy.project_id}")
    print(f"  Daily limit: {policy.daily_limit_units} units")
    print(f"  Monthly limit: {policy.monthly_limit_units} units")
    print(f"  Hard block: {policy.hard_block}")
    print(f"  Reset hour (UTC): {policy.reset_hour_utc}")
    
    # Store the policy
    budget_store.set_policy(policy)
    
    # Retrieve it
    retrieved = budget_store.get_policy("demo-project")
    print(f"\nRetrieved policy for '{retrieved.project_id}':")
    print(f"  Daily limit: {retrieved.daily_limit_units}")


def demo_2_unit_calculation() -> None:
    """Demo 2: Calculate budget units from usage."""
    print_header("Demo 2: Unit Calculation")
    
    store = BudgetStore()
    
    # Example 1: Simple row processing
    usage1 = {"rows_in": 5000, "rows_out": 2500}
    units1 = store.compute_units(usage1)
    print(f"Usage: {usage1['rows_in']:,} rows in, {usage1['rows_out']:,} rows out")
    print(f"Units: {units1} (5000/1000 + 2500/1000 = 5 + 3 = 8, rounded up)")
    
    # Example 2: With data volume
    usage2 = {"bytes_in": 25 * 1024 * 1024}  # 25 MB
    units2 = store.compute_units(usage2)
    print(f"\nUsage: {usage2['bytes_in'] / (1024*1024):.0f} MB data")
    print(f"Units: {units2} (25MB / 5MB = 5 units)")
    
    # Example 3: Operations
    usage3 = {
        "decisions": 10,              # 10 * 5 = 50 units
        "reports": 5,                 # 5 * 3 = 15 units
        "contamination_checks": 3,    # 3 * 2 = 6 units
    }
    units3 = store.compute_units(usage3)
    print(f"\nUsage:")
    print(f"  10 decisions (5 units each)")
    print(f"  5 reports (3 units each)")
    print(f"  3 contamination checks (2 units each)")
    print(f"Units: {units3} (50 + 15 + 6 = 71)")
    
    # Example 4: Full pipeline
    usage4 = {
        "rows_in": 100000,            # 100 units
        "rows_out": 50000,            # 50 units
        "bytes_in": 500 * 1024 * 1024,  # 100 units (500MB / 5MB)
        "decisions": 1,               # 5 units
        "reports": 1,                 # 3 units
        "contamination_checks": 1,    # 2 units
    }
    units4 = store.compute_units(usage4)
    print(f"\nFull pipeline example:")
    print(f"  100K rows in, 50K rows out, 500MB data")
    print(f"  1 decision, 1 report, 1 contamination check")
    print(f"Units: {units4}")


def demo_3_budget_checking() -> None:
    """Demo 3: Check budget status with simulated usage."""
    print_header("Demo 3: Budget Checking")
    
    budget_store.reset()
    
    # Create a policy with low limits for demo
    policy = BudgetPolicy(
        project_id="demo-project",
        daily_limit_units=50,
        monthly_limit_units=500,
        hard_block=True,
    )
    budget_store.set_policy(policy)
    
    # Check 1: Initial state (within budget)
    print("Check 1: No usage yet")
    result = budget_store.check_budget("demo-project", usage_store=None)
    print(f"  Mode: {result.mode}")
    print(f"  Allowed: {result.allowed}")
    print(f"  Remaining daily: {result.remaining_daily_units}")
    print(f"  Remaining monthly: {result.remaining_monthly_units}")
    print(f"  Reason: {result.reason}")
    
    # Check 2: With simulated usage (passing)
    print("\nCheck 2: 30 units used (under 50 limit)")
    
    class MockUsageStore:
        def query(self, window_minutes, group_by):
            return {
                "buckets": [{
                    "project_id": "demo-project",
                    "rows_in": 30000,  # 30 units
                }]
            }
    
    result = budget_store.check_budget("demo-project", usage_store=MockUsageStore())
    print(f"  Mode: {result.mode}")
    print(f"  Remaining daily: {result.remaining_daily_units}")
    
    # Check 3: Exceeding budget
    print("\nCheck 3: 60 units used (over 50 limit)")
    
    class MockUsageStoreExceeded:
        def query(self, window_minutes, group_by):
            return {
                "buckets": [{
                    "project_id": "demo-project",
                    "rows_in": 60000,  # 60 units (exceeds 50 limit)
                }]
            }
    
    result = budget_store.check_budget("demo-project", usage_store=MockUsageStoreExceeded())
    print(f"  Mode: {result.mode}")
    print(f"  Allowed: {result.allowed}")
    print(f"  Reason: {result.reason}")


def demo_4_soft_vs_hard_blocking() -> None:
    """Demo 4: Compare soft vs hard blocking modes."""
    print_header("Demo 4: Soft vs Hard Blocking")
    
    budget_store.reset()
    
    # Simulate exceeded usage
    class MockUsageStoreExceeded:
        def query(self, window_minutes, group_by):
            return {
                "buckets": [{
                    "project_id": "demo-project",
                    "rows_in": 15000,  # 15 units (exceeds 10 limit)
                }]
            }
    
    # Hard block policy
    hard_policy = BudgetPolicy(
        project_id="hard-project",
        daily_limit_units=10,
        hard_block=True,
    )
    budget_store.set_policy(hard_policy)
    
    result = budget_store.check_budget("hard-project", usage_store=MockUsageStoreExceeded())
    print(f"Hard block policy (exceeded):")
    print(f"  Mode: {result.mode}")
    print(f"  Allowed: {result.allowed}")
    print(f"  → Requests would be BLOCKED with HTTP 402")
    
    # Soft block policy
    soft_policy = BudgetPolicy(
        project_id="soft-project",
        daily_limit_units=10,
        hard_block=False,
    )
    budget_store.set_policy(soft_policy)
    
    result = budget_store.check_budget("soft-project", usage_store=MockUsageStoreExceeded())
    print(f"\nSoft block policy (exceeded):")
    print(f"  Mode: {result.mode}")
    print(f"  Allowed: {result.allowed}")
    print(f"  → Requests would be ALLOWED with warning header")


def demo_5_budget_status() -> None:
    """Demo 5: Get full budget status."""
    print_header("Demo 5: Full Budget Status")
    
    budget_store.reset()
    
    policy = BudgetPolicy(
        project_id="demo-project",
        daily_limit_units=1000,
        monthly_limit_units=10000,
        hard_block=True,
        reset_hour_utc=6,
    )
    budget_store.set_policy(policy)
    
    # Get full status
    status = budget_store.get_status("demo-project")
    
    print("Full budget status:")
    print_json(status)


def demo_6_config_from_file() -> None:
    """Demo 6: Load budget configuration from file."""
    print_header("Demo 6: Configuration from File")
    
    # Create a temporary config file
    config = {
        "policies": [
            {
                "project_id": "production",
                "daily_limit_units": 5000,
                "monthly_limit_units": 50000,
                "hard_block": True,
                "reset_hour_utc": 0,
            },
            {
                "project_id": "staging",
                "daily_limit_units": 500,
                "monthly_limit_units": 5000,
                "hard_block": False,
                "reset_hour_utc": 6,
            },
            {
                "project_id": "development",
                "daily_limit_units": 100,
                "monthly_limit_units": 1000,
                "hard_block": False,
                "reset_hour_utc": 12,
            },
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name
    
    try:
        # Load configuration
        store = BudgetStore()
        store.configure(config_path)
        
        print(f"Loaded configuration from: {config_path}")
        print(f"\nConfigured policies:")
        
        for project in ["production", "staging", "development"]:
            policy = store.get_policy(project)
            print(f"\n  {project}:")
            print(f"    Daily: {policy.daily_limit_units}")
            print(f"    Monthly: {policy.monthly_limit_units}")
            print(f"    Hard block: {policy.hard_block}")
            print(f"    Reset hour: {policy.reset_hour_utc}:00 UTC")
    finally:
        Path(config_path).unlink()


def demo_7_reset_time_calculation() -> None:
    """Demo 7: Calculate next reset time."""
    print_header("Demo 7: Reset Time Calculation")
    
    budget_store.reset()
    
    from datetime import datetime
    
    policy = BudgetPolicy(
        project_id="demo-project",
        daily_limit_units=100,
        reset_hour_utc=6,  # Reset at 6 AM UTC
    )
    budget_store.set_policy(policy)
    
    test_times = [
        ("2024-01-15T00:00:00", "Midnight UTC (before 6 AM)"),
        ("2024-01-15T05:59:59", "Just before reset"),
        ("2024-01-15T06:00:00", "Exactly at reset hour"),
        ("2024-01-15T12:00:00", "Noon UTC (after reset)"),
        ("2024-01-15T23:59:59", "End of day"),
    ]
    
    print(f"Reset hour: {policy.reset_hour_utc}:00 UTC")
    print()
    
    for time_str, description in test_times:
        now = datetime.fromisoformat(time_str)
        result = budget_store.check_budget("demo-project", now_utc=now)
        
        print(f"{time_str} - {description}")
        print(f"  Next reset: {result.reset_time_utc}")
        print()


def main() -> None:
    """Run all demos."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              Verifily Budget Guards Demo                     ║
    ║                                                              ║
    ║  This demo showcases the cost enforcement features of        ║
    ║  Verifily's Budget Guards v1.                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_1_basic_policy()
        demo_2_unit_calculation()
        demo_3_budget_checking()
        demo_4_soft_vs_hard_blocking()
        demo_5_budget_status()
        demo_6_config_from_file()
        demo_7_reset_time_calculation()
        
        print_header("Demo Complete")
        print("Budget Guards provides:")
        print("  ✓ Per-project budget policies")
        print("  ✓ Daily and monthly limits")
        print("  ✓ Hard and soft enforcement modes")
        print("  ✓ Unit-based cost tracking")
        print("  ✓ Configurable reset times")
        print("  ✓ Full budget status API")
        print()
        print("For more information, see: docs/budget_guards.md")
        print()
        
    except Exception as e:
        print(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
