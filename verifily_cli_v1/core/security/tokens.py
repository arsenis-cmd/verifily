"""Scoped API tokens with HMAC-SHA256 signatures.

Token format:  vft_<base64url(JSON)>.<hex HMAC-SHA256>
Prefix ``vft_`` distinguishes from API keys (``vf_`` in identity.py).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional

from verifily_cli_v1.core.security.rbac import EnterpriseRole, Permission

TOKEN_PREFIX = "vft_"


@dataclass(frozen=True)
class TokenPayload:
    """Decoded and verified token payload."""

    token_id: str
    role: EnterpriseRole
    project_id: str
    permissions_override: Optional[FrozenSet[Permission]]
    exp: float


def create_token(secret: str, payload: Dict[str, Any]) -> str:
    """Create an HMAC-signed enterprise token.

    Args:
        secret: HMAC signing secret (VERIFILY_TOKEN_SECRET).
        payload: dict with keys ``token_id``, ``role``, ``project_id``, ``exp``
                 and optional ``permissions_override`` (list of permission value strings).

    Returns:
        Token string ``vft_<b64payload>.<hex_signature>``.
    """
    json_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    b64_payload = base64.urlsafe_b64encode(json_bytes).decode().rstrip("=")
    sig = hmac.new(secret.encode(), json_bytes, hashlib.sha256).hexdigest()
    return f"{TOKEN_PREFIX}{b64_payload}.{sig}"


def verify_token(secret: str, token: str) -> TokenPayload:
    """Verify an HMAC-signed enterprise token.

    Raises:
        ValueError: on invalid format, bad signature, or expired token.
    """
    if not token.startswith(TOKEN_PREFIX):
        raise ValueError("Invalid token prefix")

    body = token[len(TOKEN_PREFIX):]
    parts = body.split(".", 1)
    if len(parts) != 2:
        raise ValueError("Invalid token format")

    b64_part, sig_part = parts

    # Re-pad base64
    padding = 4 - (len(b64_part) % 4)
    if padding != 4:
        b64_part += "=" * padding

    try:
        json_bytes = base64.urlsafe_b64decode(b64_part)
    except Exception:
        raise ValueError("Invalid base64 payload")

    # Verify HMAC (constant-time)
    expected_sig = hmac.new(secret.encode(), json_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig_part, expected_sig):
        raise ValueError("Invalid token signature")

    data = json.loads(json_bytes)

    # Check expiration
    exp = data.get("exp", 0)
    if time.time() > exp:
        raise ValueError("Token expired")

    # Parse permissions_override
    perms_override = None
    if data.get("permissions_override") is not None:
        perms_override = frozenset(Permission(p) for p in data["permissions_override"])

    return TokenPayload(
        token_id=data["token_id"],
        role=EnterpriseRole(data["role"]),
        project_id=data["project_id"],
        permissions_override=perms_override,
        exp=exp,
    )


def is_enterprise_token(token_str: str) -> bool:
    """Quick check whether a string looks like an enterprise token."""
    return token_str.startswith(TOKEN_PREFIX)
