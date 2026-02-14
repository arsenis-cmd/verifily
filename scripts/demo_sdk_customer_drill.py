#!/usr/bin/env python3
"""Verifily SDK demo — customer drill via Python SDK.

Assumes the server is already running.
Default: http://127.0.0.1:8099 (matches demo_api_customer_drill.sh)

Usage:
    python3 scripts/demo_sdk_customer_drill.py
    python3 scripts/demo_sdk_customer_drill.py --base-url http://localhost:8080
    python3 scripts/demo_sdk_customer_drill.py --api-key my-secret
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on sys.path so verifily_sdk is importable.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "verifily_sdk"))
sys.path.insert(0, REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifily SDK customer drill demo")
    parser.add_argument("--base-url", default="http://127.0.0.1:8099")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--clean-config", required=True, help="Path to clean pipeline YAML")
    parser.add_argument("--leaked-config", required=True, help="Path to leaked pipeline YAML")
    args = parser.parse_args()

    from verifily_sdk import VerifilyClient

    print()
    print("\u2554" + "\u2550" * 62 + "\u2557")
    print("\u2551  Verifily \u2014 SDK Customer Drill Demo" + " " * 26 + "\u2551")
    print("\u255a" + "\u2550" * 62 + "\u255d")
    print()

    client = VerifilyClient(base_url=args.base_url, api_key=args.api_key)
    exit_code = 0

    # ── Step 1: Health ───────────────────────────────────────────
    print("\u2501\u2501\u2501 Step 1: health() \u2501\u2501\u2501")
    h = client.health()
    print(f"  Status: {h.status}")
    print(f"  Version: {h.version}")
    if h.status != "ok":
        print("  \u2717 Health check failed")
        return 1
    print("  \u2713 Health OK")
    print()

    # ── Step 2: Pipeline clean (expect SHIP) ─────────────────────
    print("\u2501\u2501\u2501 Step 2: pipeline(clean \u2192 expect SHIP) \u2501\u2501\u2501")
    r = client.pipeline(config_path=args.clean_config, plan=True, ci=True)
    rec = r.decision.get("recommendation", "?")
    print(f"  Decision: {rec}")
    print(f"  Exit code: {r.exit_code}")
    if r.exit_code != 0:
        print("  \u2717 Expected exit_code 0 (SHIP)")
        exit_code = 1
    else:
        print("  \u2713 SHIP")
    print()

    # ── Step 3: Pipeline leaked (expect DONT_SHIP) ───────────────
    print("\u2501\u2501\u2501 Step 3: pipeline(leaked \u2192 expect DONT_SHIP) \u2501\u2501\u2501")
    r = client.pipeline(config_path=args.leaked_config, plan=True, ci=True)
    rec = r.decision.get("recommendation", "?")
    print(f"  Decision: {rec}")
    print(f"  Exit code: {r.exit_code}")
    if r.exit_code != 1:
        print("  \u2717 Expected exit_code 1 (DONT_SHIP)")
        exit_code = 1
    else:
        print("  \u2713 DONT_SHIP")
    print()

    # ── Summary ──────────────────────────────────────────────────
    if exit_code == 0:
        print("\u2554" + "\u2550" * 62 + "\u2557")
        print("\u2551  SDK Customer Drill \u2014 ALL PASSED \u2713" + " " * 27 + "\u2551")
        print("\u2560" + "\u2550" * 62 + "\u2563")
        print("\u2551  SDK CLEAN:  SHIP       exit 0" + " " * 29 + "\u2551")
        print("\u2551  SDK LEAKED: DONT_SHIP  exit 1" + " " * 29 + "\u2551")
        print("\u255a" + "\u2550" * 62 + "\u255d")
    else:
        print("  FAILED — see output above")

    client.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
