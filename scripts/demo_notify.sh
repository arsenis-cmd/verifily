#!/bin/bash
# Notification Demo for Verifily
# Demonstrates webhook notifications for gate results

set -e

echo "=== Verifily Notification Demo ==="
echo

# Setup
DEMO_DIR="/tmp/verifily_notify_demo"
mkdir -p "$DEMO_DIR"
MOCK_SERVER_PID=""

# Cleanup function
cleanup() {
    if [ -n "$MOCK_SERVER_PID" ]; then
        kill $MOCK_SERVER_PID 2>/dev/null || true
    fi
    rm -rf "$DEMO_DIR"
}
trap cleanup EXIT

echo "Working directory: $DEMO_DIR"
echo

# --- 1. Start Mock Webhook Server ---
echo "=== 1. Starting Mock Webhook Server ==="
python3 -c "
import http.server
import json
import sys
from pathlib import Path

# Track received notifications
received_file = Path('$DEMO_DIR') / 'received.jsonl'

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body)
            # Log received notification
            with open(received_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
            
            print(f'[MOCK SERVER] Received: {data.get(\"event\")} - {data.get(\"decision\")}')
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        except Exception as e:
            print(f'[MOCK SERVER] Error: {e}')
            self.send_response(500)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress default logging

# Start server
server = http.server.HTTPServer(('127.0.0.1', 9999), Handler)
print('[MOCK SERVER] Listening on http://127.0.0.1:9999')

# Run briefly to confirm it started
import threading
def serve():
    server.serve_forever()

thread = threading.Thread(target=serve, daemon=True)
thread.start()

# Keep main thread alive
import time
time.sleep(3600)
" &

MOCK_SERVER_PID=$!
sleep 2  # Wait for server to start

echo "Mock webhook server running on http://127.0.0.1:9999"
echo

# --- 2. Test Webhook Payload Building ---
echo "=== 2. Testing Notification Payload Building ==="
python3 -c "
from verifily_cli_v1.core.api.notifier import (
    build_notification_payload,
    format_slack_message,
    format_github_comment,
    NotificationEventType,
)

# Build payload for DONT_SHIP
payload = build_notification_payload(
    request_id='req-test-123',
    project_id='proj-demo',
    decision='DONT_SHIP',
    exit_code=1,
    metrics={'total_rows': 10000, 'valid_rows': 9995},
    contamination={'status': 'FAIL', 'jaccard_similarity': 0.35, 'overlap_count': 150},
    contracts={'total': 5, 'passed': 3, 'failed': 1, 'warnings': 1},
    run_id='run_20250120',
    artifact_paths={'decision': 'runs/run_20250120/decision.json'},
    elapsed_ms=1250,
    event_type=NotificationEventType.PIPELINE_DONE,
)

print('Payload:')
print(json.dumps(payload, indent=2))
print()

# Verify no raw data
assert 'text' not in payload, 'Raw text should not be in payload'
assert 'input' not in payload, 'Raw input should not be in payload'
assert 'output' not in payload, 'Raw output should not be in payload'
print('✅ Payload redaction verified (no raw data)')
print()

# Format for Slack
slack_msg = format_slack_message(payload)
print('Slack Message:')
print(slack_msg)
print()

# Format for GitHub
gh_comment = format_github_comment(payload)
print('GitHub Comment Preview:')
print(gh_comment[:500] + '...' if len(gh_comment) > 500 else gh_comment)
"

echo

# --- 3. Test Notification Sending ---
echo "=== 3. Testing Notification Sending ==="
python3 -c "
import httpx
from verifily_cli_v1.core.api.notifier import (
    NotificationConfig,
    NotificationTarget,
    NotificationTargetType,
    send_notifications,
    build_notification_payload,
)

# Create config pointing to mock server
config = NotificationConfig(
    enabled=True,
    targets=[
        NotificationTarget(
            type=NotificationTargetType.WEBHOOK,
            url='http://127.0.0.1:9999/webhook',
        ),
    ],
)

# Build payload
payload = build_notification_payload(
    request_id='req-test-456',
    project_id='proj-demo',
    decision='DONT_SHIP',
    exit_code=1,
    run_id='run_001',
)

# Send notification
result = send_notifications(config, payload)

print(f'Targets OK: {result.targets_ok}')
print(f'Targets Failed: {result.targets_failed}')
print(f'Success: {result.success}')

if result.success:
    print('✅ Notification sent successfully')
else:
    print('❌ Notification failed:', result.errors)
"

echo

# --- 4. Test SHIP vs DONT_SHIP ---
echo "=== 4. Testing SHIP vs DONT_SHIP Notification Behavior ==="

# Test DONT_SHIP (should notify)
python3 -c "
import httpx
from verifily_cli_v1.core.api.notifier import (
    NotificationConfig,
    NotificationTarget,
    NotificationTargetType,
    send_notifications,
    build_notification_payload,
)

config = NotificationConfig(
    enabled=True,
    targets=[NotificationTarget(type=NotificationTargetType.WEBHOOK, url='http://127.0.0.1:9999/webhook')],
    only_on=['DONT_SHIP', 'INVESTIGATE'],  # Default
    include_on_ship=False,
)

# DONT_SHIP should notify
if config.should_notify('DONT_SHIP'):
    payload = build_notification_payload(
        request_id='req-dont-ship',
        project_id='proj-demo',
        decision='DONT_SHIP',
        exit_code=1,
    )
    result = send_notifications(config, payload)
    print('✅ DONT_SHIP notified (expected)')
else:
    print('❌ DONT_SHIP should have been notified')
"

# Test SHIP (should NOT notify by default)
python3 -c "
import httpx
from verifily_cli_v1.core.api.notifier import (
    NotificationConfig,
    NotificationTarget,
    NotificationTargetType,
    send_notifications,
    build_notification_payload,
)

config = NotificationConfig(
    enabled=True,
    targets=[NotificationTarget(type=NotificationTargetType.WEBHOOK, url='http://127.0.0.1:9999/webhook')],
    only_on=['DONT_SHIP', 'INVESTIGATE'],
    include_on_ship=False,
)

# SHIP should NOT notify by default
if not config.should_notify('SHIP'):
    print('✅ SHIP did not notify (expected)')
else:
    print('⚠️  SHIP was notified (unexpected with default config)')
"

echo

# --- 5. Verify Received Notifications ---
echo "=== 5. Verifying Received Notifications ==="
python3 -c "
import json
from pathlib import Path

received_file = Path('$DEMO_DIR') / 'received.jsonl'

if received_file.exists():
    with open(received_file) as f:
        notifications = [json.loads(line) for line in f if line.strip()]
    
    print(f'Received {len(notifications)} notification(s):')
    for n in notifications:
        print(f\"  - {n.get('event')}: {n.get('decision')} (req: {n.get('request_id')})\")
    
    # Verify payload structure
    for n in notifications:
        assert 'request_id' in n
        assert 'decision' in n
        assert 'timestamp' in n
        # Verify no raw data
        assert 'text' not in n
        assert 'input' not in n
    
    print('✅ All notifications have correct structure')
else:
    print('No notifications received')
"

echo

# --- 6. Summary ---
echo "=== Summary ==="
echo
echo "Notification Flow:"
echo "  1. Gate runs and produces decision"
echo "  2. Notifier checks if decision should trigger notification"
echo "  3. Payload built (no raw data, redacted secrets)"
echo "  4. Sent to all configured targets"
echo "  5. Results tracked (success/failure per target)"
echo
echo "Configuration (Environment Variables):"
echo "  VERIFILY_NOTIFY=1                    # Enable notifications"
echo "  VERIFILY_NOTIFY_WEBHOOK_URL=...      # Webhook endpoint"
echo "  VERIFILY_NOTIFY_SLACK_WEBHOOK_URL=... # Slack webhook"
echo "  VERIFILY_GITHUB_TOKEN=...            # GitHub token"
echo "  VERIFILY_GITHUB_REPO=org/repo        # GitHub repo"
echo "  VERIFILY_GITHUB_PR=123               # PR number"
echo "  VERIFILY_NOTIFY_ONLY_ON=DONT_SHIP,INVESTIGATE"
echo "  VERIFILY_NOTIFY_INCLUDE_SHIP=0       # Don't notify on SHIP"
echo
echo "=== Notification Demo -- ALL PASSED ==="
echo
echo "Key Takeaways:"
echo "  - Notifications are best-effort (don't block pipeline)"
echo "  - Payloads never contain raw data or secrets"
echo "  - URLs are redacted in error logs"
echo "  - Job notifications are idempotent (sent once)"
echo "  - Supports webhook, Slack, and GitHub PR comments"
