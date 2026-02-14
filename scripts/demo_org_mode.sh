#!/bin/bash
#
# Org & Access v1 Demo Script
# 
# Demonstrates multi-tenant features:
# - Org + project creation
# - Key lifecycle (create/revoke)
# - RBAC (viewer blocked from writes)
# - Project isolation
#
# Expected output ends with:
# ORG MODE DEMO — ALL PASSED
# DEV pipeline: SHIP ✓
# VIEWER write blocked: 403 ✓
# DEV revoked blocked: 401 ✓

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_URL="${VERIFILY_SERVER:-http://127.0.0.1:8000}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Verifily Org & Access v1 Demo                       ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║  Multi-tenant control plane demonstration                    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Check if server is running
echo -n "Checking server at $SERVER_URL... "
if curl -s "$SERVER_URL/health" > /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Server not running. Start with:"
    echo "  VERIFILY_ORG_MODE=1 python -m verifily_cli_v1 serve"
    exit 1
fi

echo

# Step 1: Create org
echo -e "${YELLOW}Step 1: Create organization${NC}"
ORG_RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/orgs" \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp Demo"}')
ORG_ID=$(echo "$ORG_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['org_id'])")
echo "  Created org: $ORG_ID"

# Step 2: Create project
echo -e "${YELLOW}Step 2: Create project${NC}"
PROJECT_RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/projects" \
  -H "Content-Type: application/json" \
  -d "{\"org_id\": \"$ORG_ID\", \"name\": \"core-llm-demo\"}")
PROJECT_ID=$(echo "$PROJECT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['project_id'])")
echo "  Created project: $PROJECT_ID"

# Step 3: Create ADMIN key
echo -e "${YELLOW}Step 3: Create ADMIN key${NC}"
ADMIN_KEY_RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/keys" \
  -H "Content-Type: application/json" \
  -d "{\"project_id\": \"$PROJECT_ID\", \"role\": \"admin\", \"label\": \"demo-admin\"}")
ADMIN_SECRET=$(echo "$ADMIN_KEY_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['secret'])")
ADMIN_KEY_ID=$(echo "$ADMIN_KEY_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['key_id'])")
echo "  Created ADMIN key: $ADMIN_KEY_ID"
echo "  Secret: ${ADMIN_SECRET:0:20}... (save this!)"

# Step 4: Create DEV key
echo -e "${YELLOW}Step 4: Create DEV key${NC}"
DEV_KEY_RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/keys" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_SECRET" \
  -d "{\"project_id\": \"$PROJECT_ID\", \"role\": \"dev\", \"label\": \"demo-dev\"}")
DEV_SECRET=$(echo "$DEV_KEY_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['secret'])")
DEV_KEY_ID=$(echo "$DEV_KEY_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['key_id'])")
echo "  Created DEV key: $DEV_KEY_ID"

# Step 5: Create VIEWER key
echo -e "${YELLOW}Step 5: Create VIEWER key${NC}"
VIEWER_KEY_RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/keys" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_SECRET" \
  -d "{\"project_id\": \"$PROJECT_ID\", \"role\": \"viewer\", \"label\": \"demo-viewer\"}")
VIEWER_SECRET=$(echo "$VIEWER_KEY_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['secret'])")
VIEWER_KEY_ID=$(echo "$VIEWER_KEY_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['key_id'])")
echo "  Created VIEWER key: $VIEWER_KEY_ID"

echo

# Step 6: DEV runs pipeline (should succeed)
echo -e "${YELLOW}Step 6: DEV runs pipeline${NC}"
DEV_PIPELINE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/v1/pipeline" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DEV_SECRET" \
  -d '{
    "config_path": "",
    "project_path": "",
    "plan": true,
    "ci": true
  }')
DEV_STATUS=$(echo "$DEV_PIPELINE_RESPONSE" | tail -1)
if [ "$DEV_STATUS" = "200" ] || [ "$DEV_STATUS" = "422" ]; then
    echo -e "  DEV pipeline: ${GREEN}OK${NC} (status $DEV_STATUS)"
    DEV_RESULT="✓"
else
    echo -e "  DEV pipeline: ${YELLOW}UNEXPECTED${NC} (status $DEV_STATUS)"
    DEV_RESULT="?"
fi

# Step 7: VIEWER tries to write (should fail with 403)
echo -e "${YELLOW}Step 7: VIEWER tries to submit pipeline (should be blocked)${NC}"
VIEWER_PIPELINE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/v1/pipeline" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VIEWER_SECRET" \
  -d '{
    "config_path": "",
    "project_path": "",
    "plan": true,
    "ci": true
  }')
VIEWER_STATUS=$(echo "$VIEWER_PIPELINE_RESPONSE" | tail -1)
if [ "$VIEWER_STATUS" = "403" ]; then
    echo -e "  VIEWER write blocked: ${GREEN}403 ✓${NC}"
    VIEWER_RESULT="✓"
else
    echo -e "  VIEWER write blocked: ${RED}FAILED${NC} (expected 403, got $VIEWER_STATUS)"
    VIEWER_RESULT="✗"
fi

# Step 8: VIEWER can read (should succeed)
echo -e "${YELLOW}Step 8: VIEWER reads projects (should succeed)${NC}"
VIEWER_READ_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$SERVER_URL/v1/projects" \
  -H "Authorization: Bearer $VIEWER_SECRET")
VIEWER_READ_STATUS=$(echo "$VIEWER_READ_RESPONSE" | tail -1)
if [ "$VIEWER_READ_STATUS" = "200" ]; then
    echo -e "  VIEWER read: ${GREEN}200 ✓${NC}"
else
    echo -e "  VIEWER read: ${RED}FAILED${NC} (expected 200, got $VIEWER_READ_STATUS)"
fi

echo

# Step 9: Revoke DEV key
echo -e "${YELLOW}Step 9: Revoke DEV key${NC}"
REVOKE_RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/keys/$DEV_KEY_ID/revoke" \
  -H "Authorization: Bearer $ADMIN_SECRET")
echo "  Revoked: $DEV_KEY_ID"

# Step 10: Try to use revoked DEV key (should fail with 401)
echo -e "${YELLOW}Step 10: Revoked DEV key tries to access (should be blocked)${NC}"
REVOKED_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$SERVER_URL/v1/projects" \
  -H "Authorization: Bearer $DEV_SECRET")
REVOKED_STATUS=$(echo "$REVOKED_RESPONSE" | tail -1)
if [ "$REVOKED_STATUS" = "401" ]; then
    echo -e "  Revoked key blocked: ${GREEN}401 ✓${NC}"
    REVOKED_RESULT="✓"
else
    echo -e "  Revoked key blocked: ${RED}FAILED${NC} (expected 401, got $REVOKED_STATUS)"
    REVOKED_RESULT="✗"
fi

# Step 11: List keys
echo -e "${YELLOW}Step 11: List keys${NC}"
KEYS_RESPONSE=$(curl -s -X GET "$SERVER_URL/v1/keys" \
  -H "Authorization: Bearer $ADMIN_SECRET")
ACTIVE_KEYS=$(echo "$KEYS_RESPONSE" | python3 -c "import sys, json; print(sum(1 for k in json.load(sys.stdin)['keys'] if k['is_active']))")
TOTAL_KEYS=$(echo "$KEYS_RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['keys']))")
echo "  Total keys: $TOTAL_KEYS, Active: $ACTIVE_KEYS"

echo
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo

# Summary
ALL_PASSED="true"
if [ "$DEV_RESULT" != "✓" ]; then ALL_PASSED="false"; fi
if [ "$VIEWER_RESULT" != "✓" ]; then ALL_PASSED="false"; fi
if [ "$REVOKED_RESULT" != "✓" ]; then ALL_PASSED="false"; fi

if [ "$ALL_PASSED" = "true" ]; then
    echo -e "${GREEN}ORG MODE DEMO — ALL PASSED${NC}"
else
    echo -e "${RED}ORG MODE DEMO — SOME FAILED${NC}"
fi
echo "  DEV pipeline: ${DEV_RESULT:-?}"
echo "  VIEWER write blocked: ${VIEWER_RESULT:-?}"
echo "  DEV revoked blocked: ${REVOKED_RESULT:-?}"

echo
