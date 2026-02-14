"""Device detection, environment checks."""

from __future__ import annotations

import os
import platform
import shutil
import ssl
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class EnvCheck:
    """Result of a single environment check."""

    name: str
    status: str  # PASS, FAIL, WARN
    detail: str


def check_python_version(minimum: str = "3.9") -> EnvCheck:
    """Check Python version meets minimum requirement."""
    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    min_parts = [int(x) for x in minimum.split(".")]
    cur_parts = [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]

    ok = cur_parts[:len(min_parts)] >= min_parts
    return EnvCheck(
        name="Python version",
        status="PASS" if ok else "FAIL",
        detail=f"{current} (need >={minimum})",
    )


def check_platform() -> EnvCheck:
    """Report platform and architecture."""
    return EnvCheck(
        name="Platform",
        status="PASS",
        detail=f"{platform.system()} {platform.machine()}",
    )


def check_verifily_version() -> EnvCheck:
    """Report Verifily CLI version."""
    from verifily_cli_v1 import __version__
    return EnvCheck(
        name="Verifily version",
        status="PASS",
        detail=__version__,
    )


def check_package(name: str, import_name: str | None = None, required: bool = True) -> EnvCheck:
    """Check if a Python package is importable and report its version."""
    mod = import_name or name
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "installed")
        return EnvCheck(
            name=f"Package: {name}",
            status="PASS",
            detail=ver,
        )
    except ImportError:
        return EnvCheck(
            name=f"Package: {name}",
            status="FAIL" if required else "WARN",
            detail="Not installed" + ("" if required else " (optional)"),
        )


def check_write_permission(path: str, label: str) -> EnvCheck:
    """Check write permission for a directory."""
    try:
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".verifily_write_probe"
        probe.write_text("ok")
        probe.unlink()
        return EnvCheck(name=f"Write: {label}", status="PASS", detail=str(d))
    except Exception as e:
        return EnvCheck(
            name=f"Write: {label}",
            status="FAIL",
            detail=f"Cannot write to {path}: {e}",
        )


def check_tmp_write() -> EnvCheck:
    """Check write permission for temp directory."""
    return check_write_permission(tempfile.gettempdir(), "/tmp")


def check_cwd_write() -> EnvCheck:
    """Check write permission for current working directory."""
    return check_write_permission(".", "cwd")


def check_binary(name: str) -> EnvCheck:
    """Check if an optional binary is available on PATH."""
    path = shutil.which(name)
    if path:
        return EnvCheck(name=f"Binary: {name}", status="PASS", detail=path)
    return EnvCheck(
        name=f"Binary: {name}",
        status="WARN",
        detail="Not found (optional)",
    )


def check_ssl() -> EnvCheck:
    """Check SSL library (macOS: warn if LibreSSL)."""
    ssl_ver = ssl.OPENSSL_VERSION
    if platform.system() == "Darwin" and "LibreSSL" in ssl_ver:
        return EnvCheck(
            name="SSL library",
            status="WARN",
            detail=f"{ssl_ver} (LibreSSL — some HTTPS features may be limited)",
        )
    return EnvCheck(name="SSL library", status="PASS", detail=ssl_ver)


def check_cuda() -> EnvCheck:
    """Check CUDA availability."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            count = torch.cuda.device_count()
            return EnvCheck(
                name="CUDA",
                status="PASS",
                detail=f"{count} device(s): {device_name}",
            )
        return EnvCheck(name="CUDA", status="WARN", detail="Not available")
    except ImportError:
        return EnvCheck(name="CUDA", status="WARN", detail="PyTorch not installed")


def check_mps() -> EnvCheck:
    """Check Apple MPS availability."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return EnvCheck(name="Apple MPS", status="PASS", detail="Available")
        return EnvCheck(name="Apple MPS", status="WARN", detail="Not available")
    except ImportError:
        return EnvCheck(name="Apple MPS", status="WARN", detail="PyTorch not installed")


def check_disk_space(path: str = ".", min_gb: float = 1.0) -> EnvCheck:
    """Check available disk space."""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    status = "PASS" if free_gb >= min_gb else "FAIL"
    return EnvCheck(
        name="Disk space",
        status=status,
        detail=f"{free_gb:.1f} GB free (need >={min_gb} GB)",
    )


# Keep for backward compatibility
def check_dependency(name: str) -> EnvCheck:
    """Check if a Python package is importable."""
    return check_package(name, required=True)


def check_verifily_train() -> EnvCheck:
    """Check if verifily_train is available."""
    try:
        import verifily_train

        version = getattr(verifily_train, "__version__", "unknown")
        return EnvCheck(
            name="verifily_train",
            status="PASS",
            detail=f"v{version}",
        )
    except ImportError:
        return EnvCheck(
            name="verifily_train",
            status="WARN",
            detail="Not installed (train/eval commands unavailable)",
        )


def detect_device() -> str:
    """Detect best available compute device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_platform_info() -> Dict[str, str]:
    """Get platform information."""
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "arch": platform.machine(),
        "device": detect_device(),
    }


def check_log_format() -> EnvCheck:
    """Check logging format configuration."""
    fmt = os.environ.get("VERIFILY_LOG_FORMAT", "text")
    if fmt == "json":
        return EnvCheck(name="Log format", status="PASS", detail="json (structured)")
    return EnvCheck(name="Log format", status="WARN", detail=f"{fmt} (use json for production)")


def check_privacy_mode() -> EnvCheck:
    """Check privacy mode configuration."""
    mode = os.environ.get("VERIFILY_PRIVACY_MODE", "local")
    return EnvCheck(name="Privacy mode", status="PASS", detail=mode)


def check_auth_configured() -> EnvCheck:
    """Check if API key auth is configured."""
    if os.environ.get("VERIFILY_API_KEY"):
        return EnvCheck(name="API auth", status="PASS", detail="Configured")
    return EnvCheck(name="API auth", status="WARN", detail="Not configured (open access)")


def run_all_checks() -> List[EnvCheck]:
    """Run all environment checks and return results."""
    checks = [
        # Identity
        check_verifily_version(),
        check_python_version(),
        check_platform(),

        # Core packages (required)
        check_package("typer", required=True),
        check_package("rich", required=True),
        check_package("pyyaml", import_name="yaml", required=True),

        # API/SDK packages (optional)
        check_package("fastapi", required=False),
        check_package("uvicorn", required=False),
        check_package("pydantic", required=False),
        check_package("httpx", required=False),

        # Write permissions
        check_tmp_write(),
        check_cwd_write(),

        # Optional binaries
        check_binary("jq"),
        check_binary("docker"),

        # SSL
        check_ssl(),

        # Disk
        check_disk_space(),

        # Security
        check_log_format(),
        check_privacy_mode(),
        check_auth_configured(),
    ]
    return checks
