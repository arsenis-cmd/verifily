"""Append-only audit event log for Verifily pipeline runs.

Produces audit_log.jsonl: one JSON object per line, in chronological order.
Never includes raw dataset rows, raw PII, or secrets.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from verifily_cli_v1.core.io import ensure_dir
from verifily_cli_v1.core.secrets import redact_dict


class AuditEvent:
    """A single audit log entry."""

    __slots__ = (
        "ts", "request_id", "run_id", "project", "privacy",
        "step", "status", "exit_code", "elapsed_ms",
        "inputs", "outputs", "summary",
    )

    def __init__(
        self,
        *,
        step: str,
        status: str,
        run_id: str,
        request_id: Optional[str] = None,
        project: Optional[str] = None,
        privacy: Optional[str] = None,
        exit_code: Optional[int] = None,
        elapsed_ms: Optional[int] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ts = datetime.datetime.utcnow().isoformat() + "Z"
        self.request_id = request_id
        self.run_id = run_id
        self.project = project
        self.privacy = privacy
        self.step = step
        self.status = status
        self.exit_code = exit_code
        self.elapsed_ms = elapsed_ms
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.summary = summary or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "project": self.project,
            "privacy": self.privacy,
            "step": self.step,
            "status": self.status,
            "exit_code": self.exit_code,
            "elapsed_ms": self.elapsed_ms,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "summary": self.summary,
        }


class AuditLogger:
    """Accumulates audit events and writes audit_log.jsonl at the end.

    Usage::

        audit = AuditLogger(run_id="abc123")
        audit.start("CONTRACT")
        # ... do work ...
        audit.ok("CONTRACT", elapsed_ms=12, summary={"valid": True})
        audit.write("/path/to/output")
    """

    def __init__(
        self,
        *,
        run_id: str,
        request_id: Optional[str] = None,
        project: Optional[str] = None,
        privacy: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.request_id = request_id
        self.project = project
        self.privacy = privacy
        self._events: List[AuditEvent] = []

    def _emit(
        self,
        step: str,
        status: str,
        *,
        exit_code: Optional[int] = None,
        elapsed_ms: Optional[int] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._events.append(AuditEvent(
            step=step,
            status=status,
            run_id=self.run_id,
            request_id=self.request_id,
            project=self.project,
            privacy=self.privacy,
            exit_code=exit_code,
            elapsed_ms=elapsed_ms,
            inputs=inputs,
            outputs=outputs,
            summary=summary,
        ))

    def start(self, step: str, *, inputs: Optional[Dict[str, Any]] = None) -> None:
        self._emit(step, "START", inputs=inputs)

    def ok(
        self,
        step: str,
        *,
        elapsed_ms: Optional[int] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(step, "OK", elapsed_ms=elapsed_ms, inputs=inputs,
                   outputs=outputs, summary=summary)

    def warn(
        self,
        step: str,
        *,
        elapsed_ms: Optional[int] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(step, "WARN", elapsed_ms=elapsed_ms, summary=summary)

    def fail(
        self,
        step: str,
        *,
        exit_code: Optional[int] = None,
        elapsed_ms: Optional[int] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(step, "FAIL", exit_code=exit_code, elapsed_ms=elapsed_ms,
                   summary=summary)

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Return all events as dicts (for testing / plan mode inspection)."""
        return [e.to_dict() for e in self._events]

    def write(self, out_dir: str | Path) -> Path:
        """Write audit_log.jsonl to out_dir. Returns the file path."""
        out = ensure_dir(out_dir)
        path = out / "audit_log.jsonl"
        with open(path, "w") as f:
            for event in self._events:
                safe_event = redact_dict(event.to_dict())
                f.write(json.dumps(safe_event, separators=(",", ":")) + "\n")
        return path
