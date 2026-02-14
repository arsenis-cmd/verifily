# Status Badges

Verifily can generate Shields.io-compatible JSON badge files for displaying pipeline status.

## Generate a Badge

From a known decision:

```bash
verifily badge --decision SHIP --json
```

Output:

```json
{
  "schemaVersion": 1,
  "label": "verifily",
  "message": "SHIP",
  "color": "brightgreen"
}
```

From a run directory:

```bash
verifily badge --from-run verifily_artifacts/ --out badge.json
```

## Color Mapping

| Decision | Color | Meaning |
|----------|-------|---------|
| SHIP | brightgreen | All checks passed |
| INVESTIGATE | yellow | Risk flags, no blockers |
| DONT_SHIP | red | Blockers present |
| CONTRACT_FAIL | orange | Contract validation failed |
| TOOL_ERROR | orange | Internal error |

## Use with Shields.io

### Static badge from file

If you host `badge.json` at a public URL, use the Shields.io endpoint badge:

```markdown
![Verifily](https://img.shields.io/endpoint?url=https://your-host.com/badge.json)
```

### In CI

Generate the badge as a CI step and upload it as an artifact:

```yaml
- run: verifily badge --from-run verifily_artifacts/ --out badge.json
- uses: actions/upload-artifact@v4
  with:
    name: verifily-badge
    path: badge.json
```

### Local README

For local/private repos, include the badge JSON in your repo and reference it in documentation:

```bash
verifily badge --from-run verifily_artifacts/ --out docs/badge.json
```
