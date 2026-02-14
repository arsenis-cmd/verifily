"""Badge JSON generation for Shields.io-compatible status badges."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from verifily_cli_v1.core.io import read_json, write_json

BADGE_COLORS: Dict[str, str] = {
    "SHIP": "brightgreen",
    "INVESTIGATE": "yellow",
    "DONT_SHIP": "red",
    "CONTRACT_FAIL": "orange",
    "TOOL_ERROR": "orange",
}


def generate_badge(decision: str, label: str = "verifily") -> Dict[str, Any]:
    """Generate a Shields.io-compatible JSON badge.

    Args:
        decision: One of SHIP, DONT_SHIP, INVESTIGATE, CONTRACT_FAIL, TOOL_ERROR.
        label: Badge label (left side text).

    Returns:
        Dict with schemaVersion, label, message, color.
    """
    return {
        "schemaVersion": 1,
        "label": label,
        "message": decision,
        "color": BADGE_COLORS.get(decision, "lightgrey"),
    }


def badge_from_run(run_dir: str) -> Dict[str, Any]:
    """Load the decision from a run/artifact directory and generate a badge.

    Tries decision_summary.json first, then pipeline_result.json.

    Args:
        run_dir: Path to a run or pipeline output directory.

    Returns:
        Badge dict.

    Raises:
        FileNotFoundError: If no decision artifact is found.
    """
    d = Path(run_dir)

    # Try decision_summary.json first
    for candidate in ["decision_summary.json", "pipeline_result.json"]:
        p = d / candidate
        if p.exists():
            data = read_json(p)
            # pipeline_result.json nests decision under "decision" key
            if candidate == "pipeline_result.json" and "decision" in data:
                data = data["decision"]
            rec = data.get("recommendation", "UNKNOWN")
            return generate_badge(rec)

    raise FileNotFoundError(
        f"No decision artifact found in {run_dir}. "
        f"Expected decision_summary.json or pipeline_result.json."
    )


def write_badge(badge: Dict[str, Any], out_path: str) -> Path:
    """Write badge JSON to a file."""
    return write_json(out_path, badge)
