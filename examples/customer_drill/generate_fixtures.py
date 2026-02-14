#!/usr/bin/env python3
"""Generate all fixture data for the customer drill demo.

Run once to create/refresh static fixture files.  The generated files are
committed to the repo so this script only needs to be re-run if the data
design changes.

Usage:
    python3 examples/customer_drill/generate_fixtures.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

DRILL_DIR = Path(__file__).resolve().parent
RAW_DIR = DRILL_DIR / "raw"
RUNS_DIR = DRILL_DIR / "runs"
EXPECTED_DIR = DRILL_DIR / "expected"

# ── Ticket data ──────────────────────────────────────────────────

TICKETS = [
    # (ticket_id, subject, body, resolution, category, created_at, extra_noise)
    # 1-22: unique valid tickets
    ("T001", "Login not working",
     "I cannot log into my account. I have tried resetting my password three times but the reset email never arrives. I checked spam folders and waited over an hour.",
     "Reset password via the account settings page. If the reset email does not arrive within 15 minutes check your spam folder or contact support to verify your email address on file.",
     "authentication", "2026-01-15", "browser: chrome"),

    ("T002", "Billing discrepancy",
     "My invoice shows 99 dollars but I am on the 49 dollar plan. I upgraded last month and was told the new rate would start immediately.",
     "Adjusted invoice to reflect the correct plan amount of 49 dollars. Applied a credit of 50 dollars for the overcharge.",
     "billing", "2026-01-15", "priority: high"),

    ("T003", "Slow page load",
     "The dashboard takes over 30 seconds to load every time I navigate to it. Other pages seem fine but the main dashboard is extremely slow.",
     "Cleared CDN cache and deployed optimized bundle. Dashboard load time is now under 2 seconds.",
     "performance", "2026-01-16", ""),

    ("T004", "Export button missing",
     "The CSV export button disappeared after the last update. I rely on this feature for my weekly reports.",
     "Restored export functionality in version 2.3.1 hotfix. The button is now visible under the Reports tab.",
     "feature", "2026-01-16", "version: 2.3"),

    ("T005", "Search returns wrong results",
     "When I search for customer name the results show unrelated records. This started happening after the system update last week.",
     "Fixed the search indexing issue that was causing incorrect ranking. Rebuilt the search index with correct field weights.",
     "search", "2026-01-17", ""),

    ("T006", "Email notifications delayed",
     "I receive email notifications about new orders 3 to 4 hours late. This is causing missed SLA deadlines for our team.",
     "Identified a bottleneck in the notification queue. Increased worker count and notifications now arrive within 5 minutes.",
     "notifications", "2026-01-17", "priority: high"),

    ("T007", "Dark mode colors wrong",
     "Several UI elements are unreadable in dark mode. Button text appears black on dark gray background making it impossible to read.",
     "Updated the dark mode color palette to ensure all text elements have sufficient contrast ratio of at least 4.5 to 1.",
     "design", "2026-01-17", ""),

    ("T008", "API rate limit too low",
     "Our integration hits the rate limit within 10 minutes of batch processing. We need a higher limit for enterprise use.",
     "Increased the API rate limit for enterprise accounts from 100 to 500 requests per minute.",
     "api", "2026-01-18", "plan: enterprise"),

    ("T009", "Mobile app crashes on upload",
     "The iOS app crashes when trying to upload images larger than 10MB. Android works fine with the same images.",
     "Fixed memory management issue in the iOS image compression module. Images up to 50MB now upload correctly.",
     "mobile", "2026-01-18", "platform: ios"),

    ("T010", "Two-factor setup fails",
     "I cannot enable two-factor authentication. The QR code scanner works but the verification code is always rejected.",
     "Resolved a time synchronization issue with the TOTP server. Two-factor authentication setup now works correctly.",
     "security", "2026-01-18", ""),

    ("T011", "Report formatting broken",
     "PDF reports have overlapping text and missing charts since the last update. The reports are unusable in their current state.",
     "Fixed the PDF rendering engine layout calculations. Reports now display correctly with all charts and proper text positioning.",
     "reporting", "2026-01-19", ""),

    ("T012", "Webhook delivery failing",
     "Our webhook endpoint returns 200 but the system marks deliveries as failed. We checked our server logs and the requests arrive fine.",
     "Fixed a timeout issue where the system was not waiting long enough for the webhook response. Increased timeout to 30 seconds.",
     "integrations", "2026-01-19", ""),

    ("T013", "Permission denied on shared folders",
     "Team members cannot access shared project folders even though I have set the correct permissions. This affects five team members.",
     "Corrected a permission inheritance bug that was blocking access on nested shared folders. All team members can now access the folders.",
     "access", "2026-01-19", ""),

    ("T014", "Calendar sync not working",
     "My Google Calendar integration stopped syncing events two days ago. Disconnecting and reconnecting the integration does not help.",
     "Refreshed the OAuth tokens for the calendar integration. Events are now syncing correctly and will continue to auto-refresh.",
     "integrations", "2026-01-20", ""),

    ("T015", "Duplicate invoice generated",
     "The system generated two identical invoices for the same order. I need one of them cancelled before the customer is charged twice.",
     "Cancelled the duplicate invoice and added a deduplication check to prevent future duplicate invoice generation.",
     "billing", "2026-01-20", ""),

    ("T016", "File preview not loading",
     "Document preview shows a blank page for PDF files uploaded after January 10th. Word documents preview fine.",
     "Updated the PDF viewer component to handle the new file format. All PDF files now preview correctly regardless of upload date.",
     "documents", "2026-01-20", ""),

    ("T017", "Saved filters disappearing",
     "My custom saved filters in the analytics dashboard keep disappearing after I log out and back in.",
     "Fixed a session storage issue that was causing saved filters to be cleared on logout. Filters now persist across sessions.",
     "analytics", "2026-01-21", ""),

    ("T018", "SSO login redirect loop",
     "When trying to log in via SSO I get stuck in an infinite redirect loop between our identity provider and the app.",
     "Fixed the SAML assertion parsing to handle the updated response format from your identity provider.",
     "authentication", "2026-01-21", ""),

    ("T019", "Data import timeout",
     "Importing a CSV file with 50000 rows times out after 5 minutes. Smaller files under 10000 rows import without issue.",
     "Implemented chunked processing for large CSV imports. Files up to 500000 rows now import successfully within the timeout window.",
     "data", "2026-01-21", ""),

    ("T020", "Notification preferences reset",
     "My notification preferences keep resetting to defaults every time there is a system update or maintenance window.",
     "Fixed the migration script that was overwriting user notification preferences during system updates.",
     "notifications", "2026-01-22", ""),

    ("T021", "Dashboard chart tooltips wrong",
     "The tooltips on the revenue chart show values from the previous quarter instead of the current one.",
     "Corrected the data binding for chart tooltips to reference the correct time period in the dataset.",
     "analytics", "2026-01-22", ""),

    ("T022", "Password complexity error unclear",
     "The error message just says password is invalid without explaining what requirements are missing.",
     "Updated the password validation to show specific unmet requirements such as minimum length and special character rules.",
     "ux", "2026-01-22", ""),

    # 23-24: exact duplicates of T001 and T002 (different ticket_id, same content)
    ("T023", "Login not working",
     "I cannot log into my account. I have tried resetting my password three times but the reset email never arrives. I checked spam folders and waited over an hour.",
     "Reset password via the account settings page. If the reset email does not arrive within 15 minutes check your spam folder or contact support to verify your email address on file.",
     "authentication", "2026-01-23", "browser: firefox"),

    ("T024", "Billing discrepancy",
     "My invoice shows 99 dollars but I am on the 49 dollar plan. I upgraded last month and was told the new rate would start immediately.",
     "Adjusted invoice to reflect the correct plan amount of 49 dollars. Applied a credit of 50 dollars for the overcharge.",
     "billing", "2026-01-23", "priority: medium"),

    # 25-26: near-duplicates of T003 and T004 (small word swaps)
    ("T025", "Slow page load",
     "The dashboard takes over 30 seconds to load each time I navigate to it. Other pages seem fine but the main dashboard is very slow.",
     "Cleared CDN cache and deployed optimized bundle. Dashboard load time is now below 2 seconds.",
     "performance", "2026-01-23", ""),

    ("T026", "Export button missing",
     "The CSV export button vanished after the recent update. I depend on this feature for my weekly reports.",
     "Restored export functionality in version 2.3.1 hotfix. The button is now available under the Reports tab.",
     "feature", "2026-01-23", "version: 2.3"),

    # 27-28: rows with empty required fields (will be dropped)
    ("T027", "",
     "I have been experiencing various issues with the platform and need help.",
     "Requested more details from the customer to troubleshoot the reported issues.",
     "general", "2026-01-24", ""),

    ("T028", "System error 500",
     "I keep getting a server error 500 when trying to save my profile changes.",
     "",
     "bug", "2026-01-24", "severity: critical"),

    # 29-30: PII rows (valid content + email/phone)
    ("T029", "Account recovery needed",
     "I need to recover my account. My registered email is john.doe@example.com and I signed up in December 2025.",
     "Sent account recovery instructions to the registered email address on file.",
     "authentication", "2026-01-24", ""),

    ("T030", "Billing callback request",
     "Please call me back about my billing issue at 555-867-5309 as soon as possible.",
     "Scheduled a callback with the billing team for the next business day.",
     "billing", "2026-01-24", "priority: urgent"),
]

CSV_HEADER = ["ticket_id", "subject", "body", "resolution", "category", "created_at", "extra_noise"]


def _canonical_input(subject: str, body: str) -> str:
    """Replicate the ingest canonicalization for SFT question+context → input."""
    body = body.strip()
    subject = subject.strip()
    if body:
        return f"Context:\n{body}\n\nQuestion:\n{subject}"
    return f"Question:\n{subject}"


def _canonical_output(resolution: str) -> str:
    return resolution.strip()


def _canonical_text(subject: str, body: str, resolution: str) -> str:
    """The text that _row_text() will produce: input + ' ' + output."""
    return _canonical_input(subject, body) + " " + _canonical_output(resolution)


def _ngrams(text: str, n: int = 3):
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def write_csv():
    path = RAW_DIR / "support_tickets.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for t in TICKETS:
            w.writerow(list(t))
    print(f"  Wrote {path} ({len(TICKETS)} rows)")


def build_canonical_rows():
    """Build the canonical form of all valid rows (what ingest would produce)."""
    rows = []
    for t in TICKETS:
        tid, subject, body, resolution, category, date, noise = t
        if not subject.strip() or not resolution.strip():
            continue  # dropped by ingest
        inp = _canonical_input(subject, body)
        out = _canonical_output(resolution)
        canonical = {"input": inp, "output": out, "tags": {"source": "customer_drill"}}
        row_json = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
        sha = hashlib.sha256(row_json.encode("utf-8")).hexdigest()
        canonical["id"] = "row_" + sha[:16]
        rows.append((t, canonical))
    return rows


def write_eval_clean():
    """10 eval rows with completely different content — no overlap with training."""
    eval_rows = [
        {"input": "Context:\nThe mobile app shows an error code E-4012 when trying to attach files in the chat feature.\n\nQuestion:\nChat attachment error",
         "output": "Updated the file attachment module to handle the new API response format. Attachments now work correctly in chat."},
        {"input": "Context:\nOur team dashboard shows incorrect member count after we removed three users last week.\n\nQuestion:\nIncorrect team member count",
         "output": "Recalculated the team member count by re-syncing with the user directory. The count now reflects the correct number."},
        {"input": "Context:\nThe automated backup job has been failing silently for the past 48 hours.\n\nQuestion:\nBackup job failing",
         "output": "Identified a disk space issue on the backup server. Freed up space and restarted the backup scheduler."},
        {"input": "Context:\nUsers in the APAC region report that the application is unreachable during peak hours between 9am and 11am local time.\n\nQuestion:\nAPAC region outage",
         "output": "Added a new CDN edge node in Singapore to handle APAC traffic. Latency reduced from 3 seconds to 200 milliseconds."},
        {"input": "Context:\nThe audit log shows entries for actions that no one on our team performed.\n\nQuestion:\nUnexplained audit log entries",
         "output": "Traced the entries to an automated service account running scheduled maintenance tasks. Updated the logs to distinguish automated actions."},
        {"input": "Context:\nThe drag and drop file upload feature does not work on Safari 17.\n\nQuestion:\nSafari drag and drop broken",
         "output": "Applied a polyfill for the DataTransfer API that Safari 17 handles differently. Drag and drop now works across all browsers."},
        {"input": "Context:\nOur SSO integration breaks when users have special characters like ampersand in their display name.\n\nQuestion:\nSSO special character issue",
         "output": "Fixed XML encoding of special characters in SAML assertions. Display names with special characters now authenticate correctly."},
        {"input": "Context:\nThe scheduled report emails arrive with empty CSV attachments even though the data exists in the dashboard.\n\nQuestion:\nEmpty report attachments",
         "output": "Fixed a race condition where the CSV was generated before the data query completed. Reports now wait for data before generating attachments."},
        {"input": "Context:\nNew users cannot complete the onboarding wizard because the final step button is disabled.\n\nQuestion:\nOnboarding wizard stuck",
         "output": "Fixed a form validation bug that was incorrectly marking the terms checkbox as unchecked. The wizard now completes as expected."},
        {"input": "Context:\nThe bulk user import via CSV rejects valid email addresses that contain plus signs.\n\nQuestion:\nBulk import email validation",
         "output": "Updated the email validation regex to accept plus signs as valid characters per RFC 5321 specification."},
    ]
    path = RAW_DIR / "eval_clean.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {path} ({len(eval_rows)} rows)")


def write_eval_leaked_exact(canonical_rows):
    """10 eval rows — 4 are exact copies of training rows (T001, T005, T010, T015)."""
    # Pick 4 training rows to leak
    leak_indices = [0, 4, 9, 14]  # T001, T005, T010, T015
    leaked = []
    for idx in leak_indices:
        _, canon = canonical_rows[idx]
        leaked.append({"input": canon["input"], "output": canon["output"]})

    # 6 clean rows
    clean = [
        {"input": "Context:\nThe inventory count is off by 12 units after the last stock transfer between warehouses.\n\nQuestion:\nInventory count mismatch",
         "output": "Reconciled the inventory records with physical count. The discrepancy was caused by a delayed transfer confirmation."},
        {"input": "Context:\nCustomers report that discount codes from our holiday promotion are being rejected at checkout.\n\nQuestion:\nDiscount codes rejected",
         "output": "Reactivated the expired promotion codes and extended the redemption window by 7 days as a goodwill gesture."},
        {"input": "Context:\nThe user activity heatmap on the analytics page crashes the browser tab when viewing data for more than 30 days.\n\nQuestion:\nHeatmap browser crash",
         "output": "Implemented data sampling for large date ranges. The heatmap now renders smoothly for ranges up to 365 days."},
        {"input": "Context:\nAutomated test suite started failing after the database migration to PostgreSQL 16.\n\nQuestion:\nTest suite broken after migration",
         "output": "Updated the test database fixtures to match the new PostgreSQL 16 column ordering behavior."},
        {"input": "Context:\nThe print stylesheet is missing causing browser print to output unstyled pages with broken layouts.\n\nQuestion:\nPrint layout broken",
         "output": "Added a dedicated print CSS stylesheet that formats pages correctly for A4 and Letter paper sizes."},
        {"input": "Context:\nPush notifications on Android 14 are not appearing even though notification permissions are granted.\n\nQuestion:\nAndroid push notifications missing",
         "output": "Updated the Firebase Cloud Messaging SDK to version 24 which handles the new Android 14 notification channel requirements."},
    ]

    all_rows = leaked + clean
    path = RAW_DIR / "eval_leaked_exact.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {path} ({len(all_rows)} rows, {len(leaked)} exact leaks)")

    # Verify exact overlap
    train_texts = set()
    for _, canon in canonical_rows:
        text = canon["input"] + " " + canon["output"]
        train_texts.add(hashlib.sha256(text.encode()).hexdigest())

    found = 0
    for r in all_rows:
        text = r["input"] + " " + r["output"]
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in train_texts:
            found += 1
    print(f"    Verified: {found}/10 exact overlaps (expect 4)")
    assert found == 4, f"Expected 4 exact overlaps, got {found}"


def write_eval_leaked_near(canonical_rows):
    """10 eval rows — 3 are near-duplicates of training rows (T003, T007, T012)."""
    # Near-duplicate of T003 (Slow page load) — different from T025 too
    near1 = {
        "input": "Context:\nThe dashboard takes over 30 seconds to load whenever I navigate to it. Other pages seem fine but the main dashboard is incredibly slow.\n\nQuestion:\nSlow page load",
        "output": "Cleared CDN cache and deployed optimized bundle. Dashboard load time is now under 3 seconds.",
    }
    # Near-duplicate of T007 (Dark mode colors wrong)
    near2 = {
        "input": "Context:\nSeveral UI elements are unreadable in dark mode. Button text appears black on dark gray background making it very hard to read.\n\nQuestion:\nDark mode colors wrong",
        "output": "Updated the dark mode color palette to ensure all text elements meet a sufficient contrast ratio of at least 4.5 to 1.",
    }
    # Near-duplicate of T012 (Webhook delivery failing)
    near3 = {
        "input": "Context:\nOur webhook endpoint returns 200 but the system marks deliveries as failed. We checked our server logs and all the requests arrive correctly.\n\nQuestion:\nWebhook delivery failing",
        "output": "Fixed a timeout issue where the system was not waiting long enough for the webhook response. Increased the timeout to 30 seconds.",
    }

    near_rows = [near1, near2, near3]

    # 7 clean rows
    clean = [
        {"input": "Context:\nThe customer portal shows a blank white screen on Internet Explorer 11.\n\nQuestion:\nBlank screen on IE11",
         "output": "Added compatibility polyfills for Internet Explorer 11. The portal now loads correctly on legacy browsers."},
        {"input": "Context:\nBulk email sending fails when the recipient list exceeds 1000 addresses.\n\nQuestion:\nBulk email limit exceeded",
         "output": "Implemented batch processing for bulk emails. Lists of up to 50000 recipients are now processed in chunks of 500."},
        {"input": "Context:\nThe date picker component shows dates in US format but our users expect European date format.\n\nQuestion:\nDate format localization",
         "output": "Added locale-aware date formatting based on the user profile region setting."},
        {"input": "Context:\nDatabase connection pool exhaustion during peak traffic causing intermittent 503 errors.\n\nQuestion:\nConnection pool exhaustion",
         "output": "Increased the connection pool size from 20 to 50 and added connection recycling after 300 seconds of idle time."},
        {"input": "Context:\nThe video player does not support HLS streaming format which our CDN uses for live events.\n\nQuestion:\nHLS streaming not supported",
         "output": "Integrated an HLS compatible video player library. Live event streams now play smoothly across all browsers."},
        {"input": "Context:\nUser session tokens are not being invalidated after password change leaving old sessions active.\n\nQuestion:\nSessions not invalidated on password change",
         "output": "Added a session invalidation hook to the password change flow. All existing sessions are now terminated when password is updated."},
        {"input": "Context:\nThe autocomplete search dropdown flickers and disappears before the user can select a result.\n\nQuestion:\nSearch autocomplete flickering",
         "output": "Fixed a debounce timing issue in the search component. The dropdown now stays visible until an explicit selection or dismissal."},
    ]

    all_rows = near_rows + clean
    path = RAW_DIR / "eval_leaked_near.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {path} ({len(all_rows)} rows, {len(near_rows)} near-duplicates)")

    # Verify Jaccard similarities for near-dups
    train_source_indices = [2, 6, 11]  # T003, T007, T012
    for i, (near, tidx) in enumerate(zip(near_rows, train_source_indices)):
        _, canon = canonical_rows[tidx]
        train_text = canon["input"] + " " + canon["output"]
        eval_text = near["input"] + " " + near["output"]
        tng = _ngrams(train_text)
        eng = _ngrams(eval_text)
        sim = _jaccard(tng, eng)
        print(f"    Near-dup {i+1}: Jaccard = {sim:.4f} (need > 0.70)")
        assert sim > 0.70, f"Near-dup {i+1} Jaccard {sim:.4f} too low (need > 0.70)"

    # Verify no exact overlaps
    train_texts = set()
    for _, canon in canonical_rows:
        text = canon["input"] + " " + canon["output"]
        train_texts.add(hashlib.sha256(text.encode()).hexdigest())
    exact = 0
    for r in all_rows:
        text = r["input"] + " " + r["output"]
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in train_texts:
            exact += 1
    print(f"    Verified: {exact}/10 exact overlaps (expect 0)")
    assert exact == 0, f"Expected 0 exact overlaps, got {exact}"


def write_run(name: str, run_id: str, f1: float, exact_match: float):
    """Write a minimal run directory that passes contract check."""
    run_dir = RUNS_DIR / name

    # config.yaml
    config_yaml = (
        f"task: sft\n"
        f"base_model: google/flan-t5-base\n"
        f"seed: 42\n"
        f"data_paths:\n"
        f"  train: datasets/customer_train_artifact/dataset.jsonl\n"
        f"  test: raw/eval_clean.jsonl\n"
        f"training:\n"
        f"  num_epochs: 3\n"
        f"  batch_size: 8\n"
        f"  learning_rate: 0.0002\n"
        f"  max_seq_length: 512\n"
        f"lora:\n"
        f"  enabled: true\n"
        f"  r: 16\n"
        f"  alpha: 32\n"
        f"  dropout: 0.05\n"
        f"output:\n"
        f"  dir: runs/{name}\n"
        f"  save_adapter_only: true\n"
    )
    (run_dir / "config.yaml").write_text(config_yaml)

    # environment.json
    env = {
        "python": "3.11.7",
        "torch": "2.2.0",
        "transformers": "4.38.0",
        "peft": "0.9.0",
        "platform": "Darwin",
        "arch": "arm64",
        "device": "mps",
    }
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    # eval/eval_results.json
    eval_results = {
        "run_id": run_id,
        "test_data_path": "raw/eval_clean.jsonl",
        "num_examples": 10,
        "overall": {
            "exact_match": exact_match,
            "f1": f1,
        },
        "slices": {},
        "hard_examples": [],
        "eval_duration_seconds": 1.23,
    }
    (run_dir / "eval" / "eval_results.json").write_text(json.dumps(eval_results, indent=2) + "\n")

    # run_meta.json
    run_meta = {
        "run_id": run_id,
        "status": "completed",
        "task": "sft",
        "base_model": "google/flan-t5-base",
        "dataset_version": "customer_v1",
        "compute_mode": "local",
        "device": "mps",
        "started_at": "2026-02-01T10:00:00Z",
        "completed_at": "2026-02-01T10:02:00Z",
        "duration_seconds": 120.0,
        "metrics": {"train_loss": 0.42},
        "artifact_path": f"runs/{name}",
        "seed": 42,
        "run_name": run_id,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    # hashes.json — compute real hashes for all files
    files_to_hash = [
        "config.yaml",
        "environment.json",
        "eval/eval_results.json",
        "run_meta.json",
    ]
    file_hashes = {}
    for fname in files_to_hash:
        fpath = run_dir / fname
        h = hashlib.sha256(fpath.read_bytes()).hexdigest()
        file_hashes[fname] = h

    # chain hash
    chain_parts = [f"{k}={v}" for k, v in sorted(file_hashes.items())]
    chain_hash = hashlib.sha256("|".join(chain_parts).encode()).hexdigest()

    hashes = {"files": file_hashes, "chain_hash": chain_hash}
    (run_dir / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")

    print(f"  Wrote {run_dir}/ (f1={f1}, exact_match={exact_match})")


def write_expected():
    """Write expected decision JSON files."""
    ship = {
        "recommendation": "SHIP",
        "exit_code": 0,
        "description": "Clean pipeline: no contamination, metrics above threshold",
    }
    (EXPECTED_DIR / "expected_ship.json").write_text(json.dumps(ship, indent=2) + "\n")

    dont_ship = {
        "recommendation": "DONT_SHIP",
        "exit_code": 1,
        "description": "Leaked pipeline: exact contamination detected, blocker",
    }
    (EXPECTED_DIR / "expected_dont_ship.json").write_text(json.dumps(dont_ship, indent=2) + "\n")
    print(f"  Wrote expected/ (ship + dont_ship)")


def write_verifily_yaml():
    """Write the reference verifily.yaml config."""
    yaml_content = (
        "# Verifily customer drill pipeline config\n"
        "# Paths are relative to this file or absolute (set by scripts at runtime)\n"
        "#\n"
        "# This is a reference config. The demo scripts generate working configs\n"
        "# with absolute paths in /tmp/.\n"
        "\n"
        "# Schema and ingest mapping\n"
        "schema: sft\n"
        "ingest:\n"
        "  mapping:\n"
        "    question: subject\n"
        "    answer: resolution\n"
        "    context: body\n"
        "  tags:\n"
        "    source: customer_drill\n"
        "\n"
        "# Contamination thresholds\n"
        "contamination:\n"
        "  exact_threshold: 0.0     # any exact match = FAIL\n"
        "  near_threshold: 0.15     # >15% near-dups = WARN\n"
        "  jaccard_cutoff: 0.70\n"
        "\n"
        "# Pipeline settings (used by demo scripts)\n"
        "# run_dir, train_data, eval_data are set at runtime\n"
        "ship_if:\n"
        "  min_f1: 0.65\n"
        "  min_exact_match: 0.50\n"
        "  max_f1_regression: 0.03\n"
        "  max_pii_hits: 0\n"
    )
    path = DRILL_DIR / "verifily.yaml"
    path.write_text(yaml_content)
    print(f"  Wrote {path}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating customer drill fixtures...")
    print()

    # 1. CSV
    write_csv()

    # 2. Build canonical rows (to derive eval files)
    canonical_rows = build_canonical_rows()
    print(f"  Canonical rows: {len(canonical_rows)} (from {len(TICKETS)} tickets, 2 dropped)")

    # 3. Eval files
    write_eval_clean()
    write_eval_leaked_exact(canonical_rows)
    write_eval_leaked_near(canonical_rows)

    # 4. Run directories
    write_run("run_clean", "run_clean", f1=0.7200, exact_match=0.6000)
    write_run("run_leaked", "run_leaked", f1=0.7200, exact_match=0.6000)

    # 5. Expected outputs
    write_expected()

    # 6. Config
    write_verifily_yaml()

    print()
    print("Done. All fixtures generated successfully.")


if __name__ == "__main__":
    main()
