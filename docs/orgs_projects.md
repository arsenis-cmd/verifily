# Organizations & Projects

Minimal multi-tenant structure for team use. No database required — all state is in-memory with optional JSONL persistence.

## Overview

- **Organization**: A team or company. Has members with roles.
- **Project**: A workspace within an org. Maps to `project_id` used throughout Verifily (usage, jobs, monitor).
- **Membership**: Links an API key to an org with a role.

Single-user mode still works — if you never create orgs, everything behaves as before.

## Roles

| Role | Create Projects | Add Members | View Org |
|------|----------------|-------------|----------|
| OWNER | Yes | Yes | Yes |
| ADMIN | Yes | Yes | Yes |
| MEMBER | Yes | No | Yes |

The creator of an organization is automatically added as OWNER.

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/orgs` | Any | Create organization (caller becomes OWNER) |
| GET | `/v1/orgs` | Any | List orgs caller belongs to |
| POST | `/v1/projects` | Org member | Create project in org |
| GET | `/v1/projects` | Any | List projects (optional `?org_id=` filter) |
| POST | `/v1/orgs/{org_id}/memberships` | OWNER/ADMIN | Add member to org |
| GET | `/v1/orgs/{org_id}/memberships` | Any | List org members |

## CLI Commands

```bash
# Create an organization
verifily org-create --name "My Team"

# List your organizations
verifily org-list

# Create a project
verifily project-create --org <org_id> --name "LLM Training v2"

# List projects
verifily project-list
verifily project-list --org <org_id>
```

All commands accept `--server`, `--api-key`, and `--json` flags.

## SDK

```python
from verifily_sdk import VerifilyClient

client = VerifilyClient(base_url="http://localhost:8080", api_key="my-key")

# Create org
org = client.create_org(name="My Team")
print(org.id, org.name)

# List orgs
orgs = client.list_orgs()
for o in orgs.orgs:
    print(o.id, o.name)

# Create project
project = client.create_project(org_id=org.id, name="LLM Training v2")

# List projects
projects = client.list_projects(org_id=org.id)
```

## Backward Compatibility

- If auth is disabled (no `VERIFILY_API_KEY`), all existing endpoints work unchanged.
- Org/project endpoints work with anonymous callers — the anonymous key ID owns what it creates.
- Existing `project_id` fields on pipeline/contamination/report/jobs are unaffected.
- No existing API endpoints, CLI flags, or exit codes are modified.

## Persistence

By default, org/project data is in-memory only (resets on server restart). Persistence can be added via the same JSONL append-only pattern used by jobs and usage stores.
