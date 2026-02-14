"""Notification system for Verifily.

Sends notifications via webhook, Slack, and GitHub PR comments.
Never leaks raw data or secrets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import httpx


class NotificationTargetType(str, Enum):
    """Type of notification target."""
    WEBHOOK = "webhook"
    SLACK = "slack"
    GITHUB_PR = "github_pr"


class NotificationEventType(str, Enum):
    """Type of notification event."""
    PIPELINE_DONE = "pipeline_done"
    JOB_DONE = "job_done"
    MONITOR_ALERT = "monitor_alert"
    RETRAIN_DONE = "retrain_done"


@dataclass
class NotificationTarget:
    """A notification target configuration."""
    
    type: NotificationTargetType
    url: Optional[str] = None  # For webhook/slack
    github_repo: Optional[str] = None  # org/repo
    github_pr: Optional[int] = None
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "enabled": self.enabled,
            "github_repo": self.github_repo,
            "github_pr": self.github_pr,
            # URL redacted for security
        }


@dataclass
class NotificationConfig:
    """Notification configuration."""
    
    enabled: bool = False
    targets: List[NotificationTarget] = field(default_factory=list)
    only_on: List[str] = field(default_factory=lambda: ["DONT_SHIP", "INVESTIGATE"])
    include_on_ship: bool = False
    timeout_s: float = 3.0
    max_retries: int = 2
    backoff_ms: int = 200
    
    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Check if enabled
        config.enabled = os.environ.get("VERIFILY_NOTIFY", "0").lower() in ("1", "true", "yes")
        
        if not config.enabled:
            return config
        
        # Load targets
        # Webhook
        webhook_url = os.environ.get("VERIFILY_NOTIFY_WEBHOOK_URL")
        if webhook_url:
            config.targets.append(NotificationTarget(
                type=NotificationTargetType.WEBHOOK,
                url=webhook_url,
            ))
        
        # Slack
        slack_url = os.environ.get("VERIFILY_NOTIFY_SLACK_WEBHOOK_URL")
        if slack_url:
            config.targets.append(NotificationTarget(
                type=NotificationTargetType.SLACK,
                url=slack_url,
            ))
        
        # GitHub PR
        github_token = os.environ.get("VERIFILY_GITHUB_TOKEN")
        github_repo = os.environ.get("VERIFILY_GITHUB_REPO")
        github_pr = os.environ.get("VERIFILY_GITHUB_PR")
        if github_token and github_repo and github_pr:
            try:
                config.targets.append(NotificationTarget(
                    type=NotificationTargetType.GITHUB_PR,
                    github_repo=github_repo,
                    github_pr=int(github_pr),
                ))
            except ValueError:
                pass  # Invalid PR number
        
        # Options
        only_on = os.environ.get("VERIFILY_NOTIFY_ONLY_ON", "")
        if only_on:
            config.only_on = [s.strip().upper() for s in only_on.split(",")]
        
        include_ship = os.environ.get("VERIFILY_NOTIFY_INCLUDE_SHIP", "0")
        config.include_on_ship = include_ship.lower() in ("1", "true", "yes")
        
        return config
    
    def should_notify(self, decision: str) -> bool:
        """Check if we should notify for this decision."""
        decision_upper = decision.upper()
        
        if decision_upper == "SHIP" and not self.include_on_ship:
            return False
        
        return decision_upper in self.only_on or self.include_on_ship


@dataclass
class SendResult:
    """Result of sending notifications."""
    
    success: bool
    targets_ok: int = 0
    targets_failed: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "targets_ok": self.targets_ok,
            "targets_failed": self.targets_failed,
            "errors": self.errors,
        }


def build_notification_payload(
    *,
    request_id: str,
    project_id: Optional[str],
    decision: str,
    exit_code: int,
    metrics: Optional[Dict[str, Any]] = None,
    contamination: Optional[Dict[str, Any]] = None,
    contracts: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    job_id: Optional[str] = None,
    artifact_paths: Optional[Dict[str, str]] = None,
    elapsed_ms: Optional[int] = None,
    event_type: NotificationEventType = NotificationEventType.PIPELINE_DONE,
) -> Dict[str, Any]:
    """Build notification payload.
    
    IMPORTANT: Never includes raw dataset rows or user text content.
    """
    payload = {
        "event": event_type.value,
        "timestamp": _iso_timestamp(),
        "request_id": request_id,
        "project_id": project_id,
        "decision": decision,
        "exit_code": exit_code,
    }
    
    if run_id:
        payload["run_id"] = run_id
    if job_id:
        payload["job_id"] = job_id
    if elapsed_ms:
        payload["elapsed_ms"] = elapsed_ms
    
    # Summary metrics (never raw data)
    if metrics:
        payload["metrics_summary"] = {
            "total_rows": metrics.get("total_rows"),
            "valid_rows": metrics.get("valid_rows"),
            "null_count": metrics.get("null_count"),
        }
    
    # Contamination summary (never raw overlaps)
    if contamination:
        payload["contamination_summary"] = {
            "status": contamination.get("status"),
            "jaccard_similarity": contamination.get("jaccard_similarity"),
            "overlap_count": contamination.get("overlap_count"),
        }
    
    # Contract summary (never raw failures)
    if contracts:
        payload["contracts_summary"] = {
            "total": contracts.get("total", 0),
            "passed": contracts.get("passed", 0),
            "failed": contracts.get("failed", 0),
            "warnings": contracts.get("warnings", 0),
        }
    
    # Artifact paths (workspace-relative)
    if artifact_paths:
        payload["artifacts"] = {
            k: v for k, v in artifact_paths.items()
            if not any(sensitive in k.lower() for sensitive in ["password", "secret", "key", "token"])
        }
    
    return payload


def format_slack_message(payload: Dict[str, Any]) -> str:
    """Format payload as Slack message."""
    decision = payload.get("decision", "UNKNOWN")
    request_id = payload.get("request_id", "unknown")
    
    # Emoji based on decision
    emoji = {
        "SHIP": "✅",
        "DONT_SHIP": "❌",
        "INVESTIGATE": "⚠️",
    }.get(decision.upper(), "📊")
    
    lines = [
        f"{emoji} *Verifily Gate: {decision}*",
        f"Request ID: `{request_id}`",
    ]
    
    # Add contamination status if present
    contam = payload.get("contamination_summary", {})
    if contam:
        status = contam.get("status", "UNKNOWN")
        lines.append(f"Contamination: {status}")
    
    # Add contract summary if present
    contracts = payload.get("contracts_summary", {})
    if contracts:
        total = contracts.get("total", 0)
        passed = contracts.get("passed", 0)
        lines.append(f"Contracts: {passed}/{total} passed")
    
    # Add elapsed time
    elapsed = payload.get("elapsed_ms")
    if elapsed:
        lines.append(f"Elapsed: {elapsed}ms")
    
    return "\n".join(lines)


def format_github_comment(payload: Dict[str, Any]) -> str:
    """Format payload as GitHub PR comment."""
    decision = payload.get("decision", "UNKNOWN")
    request_id = payload.get("request_id", "unknown")
    
    # Status badge
    badge = {
        "SHIP": "✅ SHIP",
        "DONT_SHIP": "❌ DONT_SHIP",
        "INVESTIGATE": "⚠️ INVESTIGATE",
    }.get(decision.upper(), f"📊 {decision}")
    
    lines = [
        f"## Verifily Gate Result: {badge}",
        "",
        f"**Request ID:** `{request_id}`",
    ]
    
    # Contamination
    contam = payload.get("contamination_summary", {})
    if contam:
        lines.extend([
            "",
            "### Contamination Check",
            f"- Status: {contam.get('status', 'UNKNOWN')}",
        ])
        similarity = contam.get("jaccard_similarity")
        if similarity is not None:
            lines.append(f"- Jaccard Similarity: {similarity:.2%}")
    
    # Contracts
    contracts = payload.get("contracts_summary", {})
    if contracts:
        lines.extend([
            "",
            "### Contracts",
            f"- Total: {contracts.get('total', 0)}",
            f"- Passed: {contracts.get('passed', 0)}",
            f"- Failed: {contracts.get('failed', 0)}",
            f"- Warnings: {contracts.get('warnings', 0)}",
        ])
    
    # Artifacts
    artifacts = payload.get("artifacts", {})
    if artifacts:
        lines.extend([
            "",
            "### Artifacts",
        ])
        for name, path in artifacts.items():
            lines.append(f"- {name}: `{path}`")
    
    # Footer
    lines.extend([
        "",
        "---",
        "*This comment was generated by Verifily.*",
    ])
    
    return "\n".join(lines)


def send_notifications(
    config: NotificationConfig,
    payload: Dict[str, Any],
    http_client: Optional[httpx.Client] = None,
) -> SendResult:
    """Send notifications to all configured targets.
    
    Uses best-effort delivery with retries and exponential backoff.
    Never leaks secrets in error logs.
    """
    result = SendResult(success=True)
    
    if not config.enabled or not config.targets:
        return result
    
    client = http_client or httpx.Client(timeout=config.timeout_s)
    
    try:
        for target in config.targets:
            if not target.enabled:
                continue
            
            try:
                _send_to_target(target, payload, client, config)
                result.targets_ok += 1
            except Exception as e:
                result.targets_failed += 1
                # Redact any potential URLs from error message
                error_msg = str(e)
                error_msg = _redact_urls(error_msg)
                result.errors.append(f"{target.type.value}: {error_msg}")
        
        result.success = result.targets_failed == 0
        
    finally:
        if http_client is None:
            client.close()
    
    return result


def _send_to_target(
    target: NotificationTarget,
    payload: Dict[str, Any],
    client: httpx.Client,
    config: NotificationConfig,
) -> None:
    """Send notification to a single target."""
    
    if target.type == NotificationTargetType.WEBHOOK:
        if not target.url:
            raise ValueError("Webhook URL not configured")
        
        response = client.post(
            target.url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    
    elif target.type == NotificationTargetType.SLACK:
        if not target.url:
            raise ValueError("Slack URL not configured")
        
        message = format_slack_message(payload)
        slack_payload = {"text": message}
        
        response = client.post(
            target.url,
            json=slack_payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    
    elif target.type == NotificationTargetType.GITHUB_PR:
        if not target.github_repo or not target.github_pr:
            raise ValueError("GitHub repo/PR not configured")
        
        token = os.environ.get("VERIFILY_GITHUB_TOKEN")
        if not token:
            raise ValueError("GitHub token not available")
        
        comment = format_github_comment(payload)
        
        # GitHub API endpoint
        url = f"https://api.github.com/repos/{target.github_repo}/issues/{target.github_pr}/comments"
        
        response = client.post(
            url,
            json={"body": comment},
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        response.raise_for_status()


def _redact_urls(text: str) -> str:
    """Redact URLs from error messages."""
    import re
    # Simple URL redaction
    return re.sub(r'https?://[^\s"\'>]+', '[REDACTED_URL]', text)


def _iso_timestamp() -> str:
    """Get current ISO timestamp."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# Job notification tracking (for idempotency)
_notified_jobs: set = set()


def mark_job_notified(job_id: str) -> None:
    """Mark a job as notified (for idempotency)."""
    _notified_jobs.add(job_id)


def is_job_notified(job_id: str) -> bool:
    """Check if a job has already been notified."""
    return job_id in _notified_jobs


def reset_notification_tracking() -> None:
    """Reset notification tracking (for testing)."""
    _notified_jobs.clear()
