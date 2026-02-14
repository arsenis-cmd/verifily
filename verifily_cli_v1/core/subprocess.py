"""Safe subprocess wrapper for calling external tools."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RunResult:
    """Result of a subprocess call."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_module(
    module: str,
    args: List[str],
    *,
    capture: bool = True,
    timeout: Optional[int] = None,
) -> RunResult:
    """Run a Python module as a subprocess.

    Equivalent to: python -m <module> <args...>
    """
    cmd = [sys.executable, "-W", "ignore", "-m", module] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            returncode=124,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
        )
    except FileNotFoundError:
        return RunResult(
            returncode=127,
            stdout="",
            stderr=f"Python interpreter not found: {sys.executable}",
        )


def check_module_available(module: str) -> bool:
    """Check if a Python module can be imported."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False
