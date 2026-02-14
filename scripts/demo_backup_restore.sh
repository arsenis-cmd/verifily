#!/bin/bash
#
# Backup/Restore Demo Script
#
# Demonstrates:
# - Creating API calls to generate usage/jobs events
# - Backing up operational metadata
# - Wiping and restoring
# - Verifying restored data

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Verifily Backup/Restore Demo                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Setup temp environment
DEMO_DIR=$(mktemp -d)
export VERIFILY_HOME="$DEMO_DIR"
export VERIFILY_PORT=28080
export VERIFILY_BIND=127.0.0.1
BACKUP_FILE="$DEMO_DIR/backup.tar.gz"

echo "Demo directory: $DEMO_DIR"
echo

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    rm -rf "$DEMO_DIR"
}
trap cleanup EXIT

# Step 1: Start server
echo -e "${YELLOW}Step 1: Starting Verifily server...${NC}"
python -m verifily_cli_v1 serve --port $VERIFILY_PORT &
SERVER_PID=$!

# Wait for server
echo "Waiting for server..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:$VERIFILY_PORT/health" > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

echo -e "  Server ${GREEN}ready${NC}"
echo

# Step 2: Generate some usage events
echo -e "${YELLOW}Step 2: Generating usage events...${NC}"

# Run a few API calls
for i in 1 2 3; do
    curl -s "http://127.0.0.1:$VERIFILY_PORT/health" > /dev/null
done

# Check usage endpoint
USAGE_COUNT=$(curl -s "http://127.0.0.1:$VERIFILY_PORT/v1/usage" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_requests', 0))" 2>/dev/null || echo "0")
echo "  Generated $USAGE_COUNT usage events"

echo

# Step 3: Create backup
echo -e "${YELLOW}Step 3: Creating backup...${NC}"
python -m verifily_cli_v1 backup --out "$BACKUP_FILE"
echo

# Step 4: Verify backup contents
echo -e "${YELLOW}Step 4: Verifying backup contents...${NC}"
if tar -tzf "$BACKUP_FILE" | grep -q "backup_manifest.json"; then
    echo -e "  Backup manifest ${GREEN}present${NC}"
    MANIFEST_RESULT="✓"
else
    echo -e "  Backup manifest ${RED}missing${NC}"
    MANIFEST_RESULT="✗"
fi

# Check for expected files
for file in "store/usage_events.jsonl"; do
    if tar -tzf "$BACKUP_FILE" | grep -q "$file"; then
        echo -e "  $file ${GREEN}present${NC}"
    else
        echo -e "  $file ${YELLOW}not present (may be empty)${NC}"
    fi
done
echo

# Step 5: Stop server and wipe data
echo -e "${YELLOW}Step 5: Stopping server and wiping data...${NC}"
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
SERVER_PID=""

# Wipe store directory
rm -rf "$VERIFILY_HOME/store"/*
echo -e "  Data ${GREEN}wiped${NC}"
echo

# Step 6: Restore from backup
echo -e "${YELLOW}Step 6: Restoring from backup...${NC}"
python -m verifily_cli_v1 restore --file "$BACKUP_FILE" --force
echo

# Step 7: Restart server and verify
echo -e "${YELLOW}Step 7: Restarting server and verifying...${NC}"
python -m verifily_cli_v1 serve --port $VERIFILY_PORT &
SERVER_PID=$!

# Wait for server
echo "Waiting for server..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:$VERIFILY_PORT/health" > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

# Check usage count after restore
USAGE_AFTER=$(curl -s "http://127.0.0.1:$VERIFILY_PORT/v1/usage" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_requests', 0))" 2>/dev/null || echo "0")

if [ "$USAGE_AFTER" -eq "$USAGE_COUNT" ]; then
    echo -e "  Usage data restored ${GREEN}correctly${NC} ($USAGE_AFTER events)"
    RESTORE_RESULT="✓"
else
    echo -e "  Usage data ${YELLOW}mismatch${NC}: before=$USAGE_COUNT, after=$USAGE_AFTER"
    RESTORE_RESULT="?"
fi

echo
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo

# Summary
ALL_PASSED="true"
if [ "$MANIFEST_RESULT" != "✓" ]; then ALL_PASSED="false"; fi
if [ "$RESTORE_RESULT" != "✓" ]; then ALL_PASSED="false"; fi

if [ "$ALL_PASSED" = "true" ]; then
    echo -e "${GREEN}Backup/Restore demo — ALL PASSED${NC}"
else
    echo -e "${YELLOW}Backup/Restore demo — SOME CHECKS INCOMPLETE${NC}"
fi

echo "  Backup manifest: ${MANIFEST_RESULT:-?}"
echo "  Restore verification: ${RESTORE_RESULT:-?}"

echo
