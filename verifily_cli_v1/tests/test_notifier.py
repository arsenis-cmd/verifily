"""Tests for notification system.

Target: ~20 tests, runtime <0.5s
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from verifily_cli_v1.core.api.notifier import (
    NotificationConfig,
    NotificationTarget,
    NotificationTargetType,
    NotificationEventType,
    SendResult,
    build_notification_payload,
    format_slack_message,
    format_github_comment,
    send_notifications,
    mark_job_notified,
    is_job_notified,
    reset_notification_tracking,
)


class TestNotificationConfig:
    """Test notification configuration."""

    def test_default_config_disabled(self) -> None:
        """Default config has notifications disabled."""
        config = NotificationConfig()
        
        assert config.enabled is False
        assert len(config.targets) == 0

    def test_from_env_disabled(self) -> None:
        """Env loading respects VERIFILY_NOTIFY=0."""
        with patch.dict(os.environ, {"VERIFILY_NOTIFY": "0"}, clear=True):
            config = NotificationConfig.from_env()
            
            assert config.enabled is False

    def test_from_env_enabled(self) -> None:
        """Env loading enables with VERIFILY_NOTIFY=1."""
        env_vars = {
            "VERIFILY_NOTIFY": "1",
            "VERIFILY_NOTIFY_WEBHOOK_URL": "https://example.com/webhook",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = NotificationConfig.from_env()
            
            assert config.enabled is True
            assert len(config.targets) == 1
            assert config.targets[0].type == NotificationTargetType.WEBHOOK

    def test_should_notify_for_dont_ship(self) -> None:
        """Should notify for DONT_SHIP by default."""
        config = NotificationConfig(
            enabled=True,
            only_on=["DONT_SHIP", "INVESTIGATE"],
        )
        
        assert config.should_notify("DONT_SHIP") is True
        assert config.should_notify("INVESTIGATE") is True
        assert config.should_notify("SHIP") is False

    def test_should_notify_for_ship_when_enabled(self) -> None:
        """Should notify for SHIP when include_on_ship=True."""
        config = NotificationConfig(
            enabled=True,
            include_on_ship=True,
            only_on=["DONT_SHIP"],
        )
        
        assert config.should_notify("SHIP") is True


class TestBuildNotificationPayload:
    """Test payload building."""

    def test_payload_has_required_fields(self) -> None:
        """Payload contains required fields."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="DONT_SHIP",
            exit_code=1,
        )
        
        assert payload["request_id"] == "req-123"
        assert payload["project_id"] == "proj-456"
        assert payload["decision"] == "DONT_SHIP"
        assert payload["exit_code"] == 1
        assert "event" in payload
        assert "timestamp" in payload

    def test_payload_no_raw_data(self) -> None:
        """Payload never contains raw data."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="SHIP",
            exit_code=0,
            metrics={"total_rows": 1000},
        )
        
        # Should not have raw text/rows
        assert "text" not in payload
        assert "input" not in payload
        assert "output" not in payload
        
        # Should have summary
        assert "metrics_summary" in payload

    def test_payload_redacts_sensitive_paths(self) -> None:
        """Sensitive paths are redacted."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="SHIP",
            exit_code=0,
            artifact_paths={
                "decision": "runs/run_001/decision.json",
                "secret_key": "runs/run_001/secret.txt",  # Should be filtered
            },
        )
        
        assert "decision" in payload["artifacts"]
        assert "secret_key" not in payload["artifacts"]

    def test_payload_includes_contamination_summary(self) -> None:
        """Contamination summary included without raw data."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="SHIP",
            exit_code=0,
            contamination={
                "status": "PASS",
                "jaccard_similarity": 0.05,
                "overlap_count": 10,
                # Should NOT include raw overlaps
            },
        )
        
        assert payload["contamination_summary"]["status"] == "PASS"
        assert "overlap_count" in payload["contamination_summary"]


class TestFormatSlackMessage:
    """Test Slack message formatting."""

    def test_slack_message_includes_decision(self) -> None:
        """Slack message includes decision."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="DONT_SHIP",
            exit_code=1,
        )
        
        message = format_slack_message(payload)
        
        assert "DONT_SHIP" in message
        assert "req-123" in message

    def test_slack_emoji_for_ship(self) -> None:
        """SHIP gets checkmark emoji."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="SHIP",
            exit_code=0,
        )
        
        message = format_slack_message(payload)
        
        assert "✅" in message

    def test_slack_emoji_for_dont_ship(self) -> None:
        """DONT_SHIP gets X emoji."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="DONT_SHIP",
            exit_code=1,
        )
        
        message = format_slack_message(payload)
        
        assert "❌" in message


class TestFormatGitHubComment:
    """Test GitHub comment formatting."""

    def test_github_comment_includes_decision(self) -> None:
        """GitHub comment includes decision."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="INVESTIGATE",
            exit_code=2,
        )
        
        comment = format_github_comment(payload)
        
        assert "INVESTIGATE" in comment
        assert "req-123" in comment

    def test_github_comment_markdown_format(self) -> None:
        """GitHub comment uses markdown."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="SHIP",
            exit_code=0,
        )
        
        comment = format_github_comment(payload)
        
        assert "##" in comment  # Header
        assert "**" in comment  # Bold

    def test_github_comment_includes_artifacts(self) -> None:
        """GitHub comment includes artifact paths."""
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="SHIP",
            exit_code=0,
            artifact_paths={"decision": "runs/run_001/decision.json"},
        )
        
        comment = format_github_comment(payload)
        
        assert "runs/run_001/decision.json" in comment


class TestSendNotifications:
    """Test notification sending."""

    def test_send_no_targets_returns_success(self) -> None:
        """No targets means success."""
        config = NotificationConfig(enabled=True, targets=[])
        payload = {"test": "data"}
        
        result = send_notifications(config, payload)
        
        assert result.success is True
        assert result.targets_ok == 0

    def test_send_disabled_returns_success(self) -> None:
        """Disabled config returns success without sending."""
        config = NotificationConfig(enabled=False)
        payload = {"test": "data"}
        
        result = send_notifications(config, payload)
        
        assert result.success is True

    def test_send_webhook_uses_post(self) -> None:
        """Webhook sends POST request."""
        config = NotificationConfig(
            enabled=True,
            targets=[NotificationTarget(
                type=NotificationTargetType.WEBHOOK,
                url="https://example.com/webhook",
            )],
        )
        payload = {"test": "data"}
        
        # Mock transport
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)
        
        result = send_notifications(config, payload, http_client=mock_client)
        
        assert result.targets_ok == 1
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://example.com/webhook"

    def test_send_slack_formats_message(self) -> None:
        """Slack target formats message."""
        config = NotificationConfig(
            enabled=True,
            targets=[NotificationTarget(
                type=NotificationTargetType.SLACK,
                url="https://hooks.slack.com/test",
            )],
        )
        payload = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="SHIP",
            exit_code=0,
        )
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)
        
        result = send_notifications(config, payload, http_client=mock_client)
        
        assert result.targets_ok == 1
        call_args = mock_client.post.call_args
        sent_json = call_args[1]["json"]
        assert "text" in sent_json  # Slack format

    def test_send_failure_tracked(self) -> None:
        """Failed sends are tracked."""
        config = NotificationConfig(
            enabled=True,
            targets=[NotificationTarget(
                type=NotificationTargetType.WEBHOOK,
                url="https://example.com/webhook",
            )],
        )
        payload = {"test": "data"}
        
        mock_client = MagicMock()
        mock_client.post = MagicMock(side_effect=httpx.ConnectError("Connection failed"))
        
        result = send_notifications(config, payload, http_client=mock_client)
        
        assert result.success is False
        assert result.targets_failed == 1
        assert len(result.errors) == 1

    def test_send_redacts_urls_in_errors(self) -> None:
        """URLs are redacted in error messages."""
        config = NotificationConfig(
            enabled=True,
            targets=[NotificationTarget(
                type=NotificationTargetType.WEBHOOK,
                url="https://secret.com/webhook",
            )],
        )
        payload = {"test": "data"}
        
        mock_client = MagicMock()
        mock_client.post = MagicMock(side_effect=Exception("Error connecting to https://secret.com/webhook"))
        
        result = send_notifications(config, payload, http_client=mock_client)
        
        assert result.targets_failed == 1
        error_msg = result.errors[0]
        assert "https://secret.com" not in error_msg
        assert "[REDACTED_URL]" in error_msg or "secret" not in error_msg.lower()


class TestJobNotificationIdempotency:
    """Test job notification idempotency."""

    def test_mark_job_notified(self) -> None:
        """Can mark job as notified."""
        reset_notification_tracking()
        
        mark_job_notified("job-123")
        
        assert is_job_notified("job-123") is True
        assert is_job_notified("job-456") is False

    def test_reset_notification_tracking(self) -> None:
        """Can reset tracking."""
        reset_notification_tracking()
        mark_job_notified("job-123")
        
        reset_notification_tracking()
        
        assert is_job_notified("job-123") is False


class TestNotificationPayloadDeterminism:
    """Test that payload is deterministic."""

    def test_same_input_same_payload(self) -> None:
        """Same inputs produce same payload (except timestamp)."""
        payload1 = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="SHIP",
            exit_code=0,
            metrics={"total_rows": 1000},
        )
        
        payload2 = build_notification_payload(
            request_id="req-123",
            project_id="proj-456",
            decision="SHIP",
            exit_code=0,
            metrics={"total_rows": 1000},
        )
        
        # Compare all fields except timestamp
        for key in payload1:
            if key != "timestamp":
                assert payload1[key] == payload2[key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
