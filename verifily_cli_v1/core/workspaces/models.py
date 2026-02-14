"""Pydantic request/response models for workspaces endpoints.

Ws-prefixed to avoid collision with existing CreateOrgRequest etc. in core/api/models.py.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ── Requests ────────────────────────────────────────────────────

class WsCreateOrgRequest(BaseModel):
    name: str


class WsCreateProjectRequest(BaseModel):
    org_id: str
    name: str
    billing_plan: Optional[str] = "free"


class WsCreateKeyRequest(BaseModel):
    project_id: str
    role: str  # admin, editor, viewer


class WsRevokeKeyRequest(BaseModel):
    project_id: str
    api_key_id: str


# ── Responses ───────────────────────────────────────────────────

class WsCreateOrgResponse(BaseModel):
    org_id: str
    name: str


class WsCreateProjectResponse(BaseModel):
    project_id: str
    org_id: str
    name: str


class WsCreateKeyResponse(BaseModel):
    api_key: str  # returned ONCE
    api_key_id: str
    role: str


class WsRevokeKeyResponse(BaseModel):
    ok: bool


class WsMeResponse(BaseModel):
    org_id: str
    project_id: str
    role: str
    api_key_id: str
