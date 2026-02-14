"""Budget Guards v1 - Cost enforcement for Verifily.

Prevents unexpected costs by enforcing daily/monthly budgets per project.
Works with existing UsageStore for aggregation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from verifily_cli_v1.core.io import read_json

logger = logging.getLogger("verifily.budget")


class BudgetMode:
    """Budget check result modes."""
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class BudgetPolicy:
    """Budget policy for a project.
    
    Attributes:
        project_id: Project identifier
        daily_limit_units: Daily budget limit in units
        monthly_limit_units: Monthly budget limit in units
        hard_block: If True, block requests when exceeded; if False, warn only
        reset_hour_utc: Hour (0-23) when daily budget resets
    """
    project_id: str
    daily_limit_units: int = 1000
    monthly_limit_units: int = 10000
    hard_block: bool = True
    reset_hour_utc: int = 0


@dataclass
class BudgetCheckResult:
    """Result of a budget check.
    
    Attributes:
        allowed: Whether the request should proceed
        mode: PASS/WARN/BLOCK
        remaining_daily_units: Units remaining until daily limit
        remaining_monthly_units: Units remaining until monthly limit
        reason: Human-readable explanation
        reset_time_utc: ISO timestamp of next reset
    """
    allowed: bool
    mode: str
    remaining_daily_units: int
    remaining_monthly_units: int
    reason: str
    reset_time_utc: str = ""


@dataclass
class BudgetUsage:
    """Aggregated budget usage for a project."""
    project_id: str
    daily_units: int = 0
    monthly_units: int = 0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class BudgetStore:
    """Thread-safe singleton budget store.
    
    Manages budget policies and computes usage from UsageStore.
    """
    
    # Unit conversion constants
    ROWS_PER_UNIT = 1000
    BYTES_PER_UNIT = 5 * 1024 * 1024  # 5MB
    
    # Operation unit costs
    UNIT_COSTS = {
        "decision": 5,
        "report": 3,
        "contamination_check": 2,
        "classify_job": 4,
        "retrain": 50,
        "monitor_tick": 1,
    }
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._policies: Dict[str, BudgetPolicy] = {}
        self._default_policy: Optional[BudgetPolicy] = None
        self._config_path: Optional[Path] = None
    
    def configure(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load budget policies from config file or environment.
        
        Args:
            config_path: Path to JSON config file, or None to use env vars
        """
        with self._lock:
            self._config_path = Path(config_path) if config_path else None
        
        if config_path:
            self._load_from_file(config_path)
        else:
            self._load_from_env()
    
    def _load_from_file(self, path: Union[str, Path]) -> None:
        """Load policies from JSON file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Budget config not found: %s", p)
            self._load_from_env()
            return
        
        try:
            data = read_json(p)
            policies = data.get("policies", [])
            
            with self._lock:
                self._policies.clear()
                for policy_data in policies:
                    policy = BudgetPolicy(
                        project_id=policy_data["project_id"],
                        daily_limit_units=policy_data.get("daily_limit_units", 1000),
                        monthly_limit_units=policy_data.get("monthly_limit_units", 10000),
                        hard_block=policy_data.get("hard_block", True),
                        reset_hour_utc=policy_data.get("reset_hour_utc", 0),
                    )
                    self._policies[policy.project_id] = policy
            
            logger.info("Loaded %d budget policies from %s", len(policies), p)
        except Exception as e:
            logger.warning("Failed to load budget config: %s", e)
            self._load_from_env()
    
    def _load_from_env(self) -> None:
        """Load default policy from environment variables."""
        daily = int(os.environ.get("VERIFILY_BUDGET_DEFAULT_DAILY", "1000"))
        monthly = int(os.environ.get("VERIFILY_BUDGET_DEFAULT_MONTHLY", "10000"))
        hard_block = os.environ.get("VERIFILY_BUDGET_HARD_BLOCK", "1") == "1"
        reset_hour = int(os.environ.get("VERIFILY_BUDGET_RESET_HOUR", "0"))
        
        self._default_policy = BudgetPolicy(
            project_id="*",
            daily_limit_units=daily,
            monthly_limit_units=monthly,
            hard_block=hard_block,
            reset_hour_utc=reset_hour,
        )
        
        logger.info(
            "Loaded default budget policy: daily=%d, monthly=%d, hard_block=%s",
            daily, monthly, hard_block
        )
    
    def get_policy(self, project_id: str) -> BudgetPolicy:
        """Get budget policy for a project."""
        with self._lock:
            if project_id in self._policies:
                return self._policies[project_id]
            if self._default_policy:
                # Return default with project_id substituted
                return BudgetPolicy(
                    project_id=project_id,
                    daily_limit_units=self._default_policy.daily_limit_units,
                    monthly_limit_units=self._default_policy.monthly_limit_units,
                    hard_block=self._default_policy.hard_block,
                    reset_hour_utc=self._default_policy.reset_hour_utc,
                )
            # Fallback policy
            return BudgetPolicy(project_id=project_id)
    
    def set_policy(self, policy: BudgetPolicy) -> None:
        """Set a budget policy (for testing/programmatic use)."""
        with self._lock:
            self._policies[policy.project_id] = policy
    
    def compute_units(self, usage_data: Dict[str, Any]) -> int:
        """Convert usage data to budget units.
        
        Formula:
        - rows_in / 1000
        - rows_out / 1000
        - bytes_in / 5MB
        - bytes_out / 5MB
        - decisions * 5
        - reports * 3
        - contamination_checks * 2
        - classify_jobs * 4
        - retrains * 50
        - monitor_ticks * 1
        
        All rounded up to integers.
        """
        units = 0
        
        # Data volume units
        rows_in = usage_data.get("rows_in", 0)
        rows_out = usage_data.get("rows_out", 0)
        bytes_in = usage_data.get("bytes_in", 0)
        bytes_out = usage_data.get("bytes_out", 0)
        
        units += math.ceil(rows_in / self.ROWS_PER_UNIT)
        units += math.ceil(rows_out / self.ROWS_PER_UNIT)
        units += math.ceil(bytes_in / self.BYTES_PER_UNIT)
        units += math.ceil(bytes_out / self.BYTES_PER_UNIT)
        
        # Operation units
        units += usage_data.get("decisions", 0) * self.UNIT_COSTS["decision"]
        units += usage_data.get("reports", 0) * self.UNIT_COSTS["report"]
        units += usage_data.get("contamination_checks", 0) * self.UNIT_COSTS["contamination_check"]
        units += usage_data.get("classify_jobs", 0) * self.UNIT_COSTS["classify_job"]
        units += usage_data.get("retrains", 0) * self.UNIT_COSTS["retrain"]
        units += usage_data.get("monitor_ticks", 0) * self.UNIT_COSTS["monitor_tick"]
        
        return units
    
    def _calculate_reset_time(self, policy: BudgetPolicy, now: datetime) -> str:
        """Calculate next daily reset time."""
        reset_time = now.replace(hour=policy.reset_hour_utc, minute=0, second=0, microsecond=0)
        if reset_time <= now:
            reset_time += timedelta(days=1)
        return reset_time.isoformat() + "Z"
    
    def _seconds_until_reset(self, policy: BudgetPolicy, now: datetime) -> int:
        """Calculate seconds until next daily reset."""
        reset_time = now.replace(hour=policy.reset_hour_utc, minute=0, second=0, microsecond=0)
        if reset_time <= now:
            reset_time += timedelta(days=1)
        return int((reset_time - now).total_seconds())
    
    def check_budget(
        self,
        project_id: str,
        usage_store=None,
        now_utc: Optional[datetime] = None,
    ) -> BudgetCheckResult:
        """Check if project is within budget.
        
        Args:
            project_id: Project to check
            usage_store: UsageStore instance to query (optional)
            now_utc: Current time (for testing)
            
        Returns:
            BudgetCheckResult with allowance status and remaining units
        """
        policy = self.get_policy(project_id)
        now = now_utc or datetime.utcnow()
        
        # Get usage from usage_store if available
        if usage_store:
            try:
                # Query windowed usage
                daily_usage = self._get_usage_for_window(usage_store, project_id, minutes=24*60)
                monthly_usage = self._get_usage_for_window(usage_store, project_id, minutes=30*24*60)
                
                daily_units = self.compute_units(daily_usage)
                monthly_units = self.compute_units(monthly_usage)
            except Exception as e:
                logger.warning("Failed to get usage for budget check: %s", e)
                daily_units = 0
                monthly_units = 0
        else:
            daily_units = 0
            monthly_units = 0
        
        remaining_daily = max(0, policy.daily_limit_units - daily_units)
        remaining_monthly = max(0, policy.monthly_limit_units - monthly_units)
        
        reset_time = self._calculate_reset_time(policy, now)
        
        # Determine mode
        if daily_units >= policy.daily_limit_units or monthly_units >= policy.monthly_limit_units:
            # Exceeded
            if policy.hard_block:
                return BudgetCheckResult(
                    allowed=False,
                    mode=BudgetMode.BLOCK,
                    remaining_daily_units=0,
                    remaining_monthly_units=0,
                    reason=f"Budget exceeded: daily={daily_units}/{policy.daily_limit_units}, "
                           f"monthly={monthly_units}/{policy.monthly_limit_units}",
                    reset_time_utc=reset_time,
                )
            else:
                return BudgetCheckResult(
                    allowed=True,
                    mode=BudgetMode.WARN,
                    remaining_daily_units=remaining_daily,
                    remaining_monthly_units=remaining_monthly,
                    reason=f"Budget warning: daily={daily_units}/{policy.daily_limit_units}, "
                           f"monthly={monthly_units}/{policy.monthly_limit_units}",
                    reset_time_utc=reset_time,
                )
        
        # Check if approaching limits (80% threshold)
        daily_pct = daily_units / policy.daily_limit_units if policy.daily_limit_units > 0 else 0
        monthly_pct = monthly_units / policy.monthly_limit_units if policy.monthly_limit_units > 0 else 0
        
        if daily_pct >= 0.8 or monthly_pct >= 0.8:
            mode = BudgetMode.WARN
            reason = f"Budget warning: daily={daily_pct:.0%}, monthly={monthly_pct:.0%}"
        else:
            mode = BudgetMode.PASS
            reason = "Within budget"
        
        return BudgetCheckResult(
            allowed=True,
            mode=mode,
            remaining_daily_units=remaining_daily,
            remaining_monthly_units=remaining_monthly,
            reason=reason,
            reset_time_utc=reset_time,
        )
    
    def _get_usage_for_window(self, usage_store, project_id: str, minutes: int) -> Dict[str, Any]:
        """Query usage store for a time window."""
        try:
            result = usage_store.query(window_minutes=minutes, group_by="project")
            
            # Find bucket for this project
            for bucket in result.get("buckets", []):
                if bucket.get("project_id") == project_id:
                    return {
                        "rows_in": bucket.get("rows_in", 0),
                        "rows_out": bucket.get("rows_out", 0),
                        "bytes_in": bucket.get("bytes_in", 0),
                        "bytes_out": bucket.get("bytes_out", 0),
                        "decisions": bucket.get("decisions_ship", 0) + 
                                     bucket.get("decisions_dont_ship", 0) +
                                     bucket.get("decisions_investigate", 0),
                    }
            
            return {}
        except Exception:
            return {}
    
    def get_status(self, project_id: str, usage_store=None) -> Dict[str, Any]:
        """Get full budget status for a project."""
        policy = self.get_policy(project_id)
        check = self.check_budget(project_id, usage_store)
        
        now = datetime.utcnow()
        seconds_until_reset = self._seconds_until_reset(policy, now)
        
        return {
            "project_id": project_id,
            "policy": {
                "daily_limit_units": policy.daily_limit_units,
                "monthly_limit_units": policy.monthly_limit_units,
                "hard_block": policy.hard_block,
                "reset_hour_utc": policy.reset_hour_utc,
            },
            "usage": {
                "mode": check.mode,
                "daily_used": policy.daily_limit_units - check.remaining_daily_units,
                "daily_remaining": check.remaining_daily_units,
                "daily_limit": policy.daily_limit_units,
                "monthly_used": policy.monthly_limit_units - check.remaining_monthly_units,
                "monthly_remaining": check.remaining_monthly_units,
                "monthly_limit": policy.monthly_limit_units,
            },
            "next_reset": check.reset_time_utc,
            "seconds_until_reset": seconds_until_reset,
        }
    
    def reset(self) -> None:
        """Clear all policies (for test isolation)."""
        with self._lock:
            self._policies.clear()
            self._default_policy = None
            self._config_path = None


# Singleton instance
budget_store = BudgetStore()


def configure_budgets(config_path: Optional[Union[str, Path]] = None) -> None:
    """Convenience function to configure budget store."""
    budget_store.configure(config_path)


def check_budget(project_id: str, usage_store=None) -> BudgetCheckResult:
    """Convenience function to check budget."""
    return budget_store.check_budget(project_id, usage_store)
