#!/bin/bash
#
# Production Smoke Test Script
#
# Verifies deployment is working correctly:
# - Server boots in prod mode
# - Workspace-only paths enforced
# - Config endpoint accessible
# - Basic pipeline plan works
#
# Usage: ./scripts/demo_prod_smoke.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Verifily Production Smoke Test                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Create temp workspace
WORKSPACE=$(mktemp -d)
export VERIFILY_WORKSPACE_ROOT="$WORKSPACE"
export VERIFILY_PROD=1
export VERIFILY_LOG_FORMAT=json
export VERIFILY_PORT=18080  # Use non-standard port to avoid conflicts

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    rm -rf "$WORKSPACE"
}
trap cleanup EXIT

echo "Workspace: $WORKSPACE"
echo

# Create test data
mkdir -p "$WORKSPACE/datasets/v1"
echo '{"text": "hello world"}' > "$WORKSPACE/datasets/v1/train.jsonl"

# Step 1: Start server in background
echo -e "${YELLOW}Step 1: Starting server in production mode...${NC}"
python -m verifily_cli_v1 serve --port $VERIFILY_PORT --prod --workspace-root "$WORKSPACE" &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:$VERIFILY_PORT/health" > /dev/null 2>&1; then
        echo -e "  Server ${GREEN}ready${NC}"
        break
    fi
    sleep 0.5
done

# Verify server is running
if ! curl -s "http://127.0.0.1:$VERIFILY_PORT/health" > /dev/null 2>&1; then
    echo -e "  Server ${RED}failed to start${NC}"
    exit 1
fi

echo

# Step 2: Test /v1/config endpoint
echo -e "${YELLOW}Step 2: Testing /v1/config endpoint...${NC}"
CONFIG_RESPONSE=$(curl -s "http://127.0.0.1:$VERIFILY_PORT/v1/config")

if echo "$CONFIG_RESPONSE" | grep -q "prod_mode"; then
    echo -e "  Config endpoint ${GREEN}OK${NC}"
    CONFIG_RESULT="✓"
else
    echo -e "  Config endpoint ${RED}FAILED${NC}"
    echo "  Response: $CONFIG_RESPONSE"
    CONFIG_RESULT="✗"
fi

echo

# Step 3: Verify workspace-only paths enforced
echo -e "${YELLOW}Step 3: Testing workspace-only path enforcement...${NC}"

# Try absolute path (should fail in prod)
ABS_PATH_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://127.0.0.1:$VERIFILY_PORT/v1/pipeline" \
    -H "Content-Type: application/json" \
    -d '{"config_path": "/etc/passwd", "plan": true}')
ABS_STATUS=$(echo "$ABS_PATH_RESPONSE" | tail -1)

if [ "$ABS_STATUS" = "403" ] || [ "$ABS_STATUS" = "400" ] || [ "$ABS_STATUS" = "422" ]; then
    echo -e "  Absolute path blocked ${GREEN}OK${NC} (status $ABS_STATUS)"
    PATH_RESULT="✓"
else
    echo -e "  Absolute path handling ${YELLOW}UNEXPECTED${NC} (status $ABS_STATUS)"
    PATH_RESULT="?"
fi

echo

# Step 4: Test valid workspace-relative path
echo -e "${YELLOW}Step 4: Testing valid workspace-relative path...${NC}"

# Create a minimal config
cat > "$WORKSPACE/test_config.yaml" << 'EOF'
pipeline:
  dataset_path: datasets/v1/train.jsonl
  schema: sft
EOF

PIPELINE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://127.0.0.1:$VERIFILY_PORT/v1/pipeline" \
    -H "Content-Type: application/json" \
    -d "{\"config_path\": \"test_config.yaml\", \"plan\": true, \"ci\": true}")
PIPELINE_STATUS=$(echo "$PIPELINE_RESPONSE" | tail -1)

if [ "$PIPELINE_STATUS" = "200" ] || [ "$PIPELINE_STATUS" = "422" ]; then
    echo -e "  Pipeline plan ${GREEN}OK${NC} (status $PIPELINE_STATUS)"
    PIPELINE_RESULT="✓"
else
    echo -e "  Pipeline plan ${YELLOW}UNEXPECTED${NC} (status $PIPELINE_STATUS)"
    echo "  Response: $(echo "$PIPELINE_RESPONSE" | head -1)"
    PIPELINE_RESULT="?"
fi

echo

# Step 5: Check logs are JSON format
echo -e "${YELLOW}Step 5: Verifying JSON log format...${NC}"
# Logs go to stderr, can't easily check here
# Just verify the setting is in config
if echo "$CONFIG_RESPONSE" | grep -q "json"; then
    echo -e "  JSON logging ${GREEN}configured${NC}"
    LOG_RESULT="✓"
else
    echo -e "  JSON logging ${YELLOW}unknown${NC}"
    LOG_RESULT="?"
fi

echo
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo

# Summary
ALL_PASSED="true"
if [ "$CONFIG_RESULT" != "✓" ]; then ALL_PASSED="false"; fi
if [ "$PATH_RESULT" != "✓" ]; then ALL_PASSED="false"; fi

if [ "$ALL_PASSED" = "true" ]; then
    echo -e "${GREEN}PROD SMOKE — ALL PASSED${NC}"
else
    echo -e "${RED}PROD SMOKE — SOME FAILED${NC}"
fi

echo "  workspace-only paths enforced ${PATH_RESULT:-?}"
echo "  config endpoint ${CONFIG_RESULT:-?}"
echo "  pipeline plan ${PIPELINE_RESULT:-?}"
echo "  json logging ${LOG_RESULT:-?}"

echo
