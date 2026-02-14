# Workspaces: Multi-Project Onboarding

Opt-in multi-tenant workspace layer for Verifily API. Activated via `VERIFILY_WORKSPACES_ENABLED=1`.

## Quick Start

```bash
# 1. Start server with workspaces enabled
export VERIFILY_WORKSPACES_ENABLED=1
export VERIFILY_BOOTSTRAP_TOKEN=my-bootstrap-secret
verifily serve

# 2. Create first org (uses bootstrap token)
verifily ws-org-create --name "Acme Corp" --bootstrap-token my-bootstrap-secret

# 3. Create admin key (use the returned org_id)
# First, create a project and key directly via API:
curl -X POST http://localhost:8080/v1/orgs \
  -H "X-Bootstrap-Token: my-bootstrap-secret" \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp"}'

# Use the admin key for subsequent operations
verifily ws-project-create --org org_xxx --name "QA Project" -k <admin_key>
verifily ws-key-create --project proj_xxx --role editor -k <admin_key>
verifily ws-me -k <editor_key>
```

## Roles

| Role | Can Create Keys | Can Run Pipelines | Can View Usage | Can GET /v1/me |
|------|----------------|-------------------|----------------|----------------|
| `admin` | Y | Y | Y | Y |
| `editor` | - | Y | Y | Y |
| `viewer` | - | - | Y | Y |

Role mapping to internal RBAC: `admin` = ADMIN, `editor` = DEV, `viewer` = VIEWER.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `VERIFILY_WORKSPACES_ENABLED` | Yes | Set to `1` to enable |
| `VERIFILY_BOOTSTRAP_TOKEN` | No | Token for first org creation (if empty, first org creation is open) |
| `VERIFILY_WORKSPACES_STORE_PATH` | No | Path to JSON store file (defaults to temp directory) |
| `VERIFILY_KEY_SALT` | No | Salt for key hashing |

## API Endpoints

### POST /v1/orgs

Create an organization. First org can use bootstrap token (`X-Bootstrap-Token` header). Subsequent orgs require admin auth.

```json
{"name": "Acme Corp"}
```
Response: `{"org_id": "org_xxx", "name": "Acme Corp"}`

### POST /v1/projects

Create a project within an org. Requires admin role.

```json
{"org_id": "org_xxx", "name": "QA", "billing_plan": "free"}
```
Response: `{"project_id": "proj_xxx", "org_id": "org_xxx", "name": "QA"}`

### POST /v1/keys

Create an API key for a project. Requires admin role.

```json
{"project_id": "proj_xxx", "role": "editor"}
```
Response: `{"api_key": "vf_...", "api_key_id": "abc123def456", "role": "editor"}`

The `api_key` is returned **once** and never stored in plaintext.

### POST /v1/keys/revoke

Revoke an API key. Requires admin role.

```json
{"project_id": "proj_xxx", "api_key_id": "abc123def456"}
```
Response: `{"ok": true}`

### GET /v1/me

Return identity of the calling key. Any role can access.

Response: `{"org_id": "org_xxx", "project_id": "proj_xxx", "role": "editor", "api_key_id": "abc123def456"}`

## CLI Commands

```
verifily ws-org-create     --name "Acme" [--bootstrap-token ...]
verifily ws-project-create --org <org_id> --name "QA" [--plan pro]
verifily ws-key-create     --project <project_id> --role admin
verifily ws-key-revoke     --project <project_id> --key-id <key_id>
verifily ws-me
```

All commands accept `--server`, `--api-key`, and `--json` flags.

## SDK Methods

```python
from verifily_sdk import VerifilyClient

client = VerifilyClient(base_url="http://localhost:8080", api_key="vf_...")

client.ws_create_org(name="Acme", bootstrap_token="...")
client.ws_create_project(org_id="org_xxx", name="QA", billing_plan="free")
client.ws_create_key(project_id="proj_xxx", role="editor")
client.ws_revoke_key(project_id="proj_xxx", api_key_id="abc123")
client.ws_me()
```

## Storage

Data is persisted to a JSON file with nested schema:
```
orgs → projects → keys
```

Keys are hashed with SHA-256 (salted) and never stored in plaintext. File writes use atomic `os.replace()` to prevent corruption.

## Backward Compatibility

- When `VERIFILY_WORKSPACES_ENABLED=0` (default), all workspace features are invisible
- Existing auth modes (simple, advanced, org_mode, teams) are unaffected
- No existing tests or endpoints are broken
