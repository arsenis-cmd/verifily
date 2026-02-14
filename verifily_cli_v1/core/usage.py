"""Usage metering for Verifily pipeline runs.

Produces usage.json: billable units + per-step timings.
Deterministic except for wall-clock timestamps (which are excluded).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class UsageMeter:
    """Accumulate billable usage counters and per-step timings.

    Usage::

        meter = UsageMeter(run_id="abc", mode="cli", ci=True)
        meter.record_contract(valid=True, elapsed_ms=5)
        meter.record_report(rows_in=100, bytes_in=4096, elapsed_ms=12)
        meter.record_contamination(status="PASS", checked_rows=100, elapsed_ms=8)
        meter.record_decision(decision="SHIP", exit_code=0, elapsed_ms=2)
        meter.finalize(total_elapsed_ms=30)
        usage_dict = meter.to_dict()
    """

    def __init__(
        self,
        *,
        run_id: str,
        request_id: Optional[str] = None,
        mode: str = "cli",
        ci: bool = False,
        privacy: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.request_id = request_id
        self.mode = mode
        self.ci = ci
        self.privacy = privacy
        self.project_id = project_id

        # Billable units
        self._rows_in: int = 0
        self._rows_out: int = 0
        self._bytes_in: int = 0
        self._bytes_out: int = 0
        self._contamination_checks: int = 0
        self._reports_generated: int = 0
        self._decisions_generated: int = 0
        self._contracts_validated: int = 0

        # Per-step timing (ms)
        self._timing: Dict[str, int] = {}
        self._total_ms: int = 0

        # Result
        self._decision: Optional[str] = None
        self._exit_code: Optional[int] = None
        self._contamination_status: Optional[str] = None
        self._contract_valid: Optional[bool] = None

    def record_contract(self, *, valid: bool, elapsed_ms: int) -> None:
        self._contracts_validated += 1
        self._contract_valid = valid
        self._timing["contract"] = elapsed_ms

    def record_report(
        self,
        *,
        rows_in: int,
        bytes_in: int,
        elapsed_ms: int,
    ) -> None:
        self._reports_generated += 1
        self._rows_in += rows_in
        self._bytes_in += bytes_in
        self._timing["report"] = elapsed_ms

    def record_contamination(
        self,
        *,
        status: str,
        checked_rows: int,
        elapsed_ms: int,
    ) -> None:
        self._contamination_checks += 1
        self._contamination_status = status
        self._rows_out += checked_rows
        self._timing["contamination"] = elapsed_ms

    def record_decision(
        self,
        *,
        decision: str,
        exit_code: int,
        elapsed_ms: int,
    ) -> None:
        self._decisions_generated += 1
        self._decision = decision
        self._exit_code = exit_code
        self._timing["decision"] = elapsed_ms

    def finalize(self, *, total_elapsed_ms: int) -> None:
        self._total_ms = total_elapsed_ms
        self._timing["total"] = total_elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "mode": self.mode,
            "ci": self.ci,
            "privacy": self.privacy,
            "project_id": self.project_id,
            "billable_units": {
                "rows_in": self._rows_in,
                "rows_out": self._rows_out,
                "bytes_in": self._bytes_in,
                "bytes_out": self._bytes_out,
                "contamination_checks": self._contamination_checks,
                "reports_generated": self._reports_generated,
                "decisions_generated": self._decisions_generated,
                "contracts_validated": self._contracts_validated,
            },
            "timing_ms": dict(self._timing),
            "result": {
                "decision": self._decision,
                "exit_code": self._exit_code,
                "contamination_status": self._contamination_status,
                "contract_valid": self._contract_valid,
            },
        }
