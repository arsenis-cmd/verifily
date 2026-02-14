"""Tests for enterprise scoped HMAC tokens."""

from __future__ import annotations

import time

import pytest

from verifily_cli_v1.core.security.rbac import EnterpriseRole, Permission
from verifily_cli_v1.core.security.tokens import (
    TOKEN_PREFIX,
    create_token,
    is_enterprise_token,
    verify_token,
)

SECRET = "test-secret-key-1234"


def _payload(**overrides):
    base = {
        "token_id": "tok_1",
        "role": "owner",
        "project_id": "proj1",
        "exp": time.time() + 3600,
    }
    base.update(overrides)
    return base


class TestCreateToken:
    def test_has_vft_prefix(self):
        tok = create_token(SECRET, _payload())
        assert tok.startswith(TOKEN_PREFIX)

    def test_has_dot_separator(self):
        tok = create_token(SECRET, _payload())
        body = tok[len(TOKEN_PREFIX):]
        assert "." in body
        parts = body.split(".", 1)
        assert len(parts) == 2
        assert len(parts[1]) == 64  # SHA256 hex


class TestRoundtrip:
    def test_valid_roundtrip(self):
        tok = create_token(SECRET, _payload())
        payload = verify_token(SECRET, tok)
        assert payload.token_id == "tok_1"
        assert payload.role == EnterpriseRole.OWNER
        assert payload.project_id == "proj1"
        assert payload.permissions_override is None

    def test_permissions_override_roundtrip(self):
        tok = create_token(
            SECRET,
            _payload(permissions_override=["view_usage", "export_audit"]),
        )
        payload = verify_token(SECRET, tok)
        assert payload.permissions_override == frozenset({
            Permission.VIEW_USAGE,
            Permission.EXPORT_AUDIT,
        })


class TestVerifyRejects:
    def test_expired_rejected(self):
        tok = create_token(SECRET, _payload(exp=time.time() - 10))
        with pytest.raises(ValueError, match="expired"):
            verify_token(SECRET, tok)

    def test_wrong_secret(self):
        tok = create_token(SECRET, _payload())
        with pytest.raises(ValueError, match="signature"):
            verify_token("wrong-secret", tok)

    def test_tampered_payload(self):
        tok = create_token(SECRET, _payload())
        # Flip a character in the base64 payload
        body = tok[len(TOKEN_PREFIX):]
        b64, sig = body.split(".", 1)
        tampered_b64 = b64[:-1] + ("A" if b64[-1] != "A" else "B")
        tampered_tok = f"{TOKEN_PREFIX}{tampered_b64}.{sig}"
        with pytest.raises(ValueError):
            verify_token(SECRET, tampered_tok)

    def test_tampered_signature(self):
        tok = create_token(SECRET, _payload())
        tampered = tok[:-1] + ("a" if tok[-1] != "a" else "b")
        with pytest.raises(ValueError, match="signature"):
            verify_token(SECRET, tampered)

    def test_missing_prefix(self):
        tok = create_token(SECRET, _payload())
        no_prefix = tok[len(TOKEN_PREFIX):]
        with pytest.raises(ValueError, match="prefix"):
            verify_token(SECRET, no_prefix)


class TestIsEnterpriseToken:
    def test_enterprise_token(self):
        tok = create_token(SECRET, _payload())
        assert is_enterprise_token(tok)

    def test_regular_key(self):
        assert not is_enterprise_token("vf_some_api_key")

    def test_empty(self):
        assert not is_enterprise_token("")
