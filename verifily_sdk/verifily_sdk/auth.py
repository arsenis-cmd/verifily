"""API key handling for the Verifily SDK."""

from __future__ import annotations

import os
from typing import Dict, Optional


def build_auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Return Authorization header dict if an API key is available.

    Precedence: explicit api_key > VERIFILY_API_KEY env var.
    Returns empty dict if no key is configured.
    """
    key = api_key or os.environ.get("VERIFILY_API_KEY")
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}
