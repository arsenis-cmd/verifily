# Commands to Monitor Synthetic Data Generation

## 1. Check if process is running
```bash
ps aux | grep "src.synthesize" | grep -v grep
```
**What to look for**: Should show Python process with high CPU usage (90-100%)

---

## 2. See live progress (recommended)
```bash
tail -f synthetic_generation.log
```
**What to look for**: Progress bar showing "Generating synthetic examples: X%"
**Exit**: Press Ctrl+C

---

## 3. See last 50 lines of log
```bash
tail -50 synthetic_generation.log
```
**What to look for**: Latest progress updates and filter statistics

---

## 4. Count how many examples generated so far
```bash
wc -l data/synthetic/synthetic_train.jsonl 2>/dev/null || echo "File not created yet"
```
**What to look for**: Number should increase toward 100,000

---

## 5. Check file size (growing)
```bash
ls -lh data/synthetic/synthetic_train.jsonl 2>/dev/null || echo "File not created yet"
```
**What to look for**: File size increasing (final ~100-150MB)

---

## 6. See full status summary
```bash
echo "=== Process Status ===" && \
ps aux | grep "src.synthesize" | grep -v grep && \
echo "" && \
echo "=== Generated So Far ===" && \
wc -l data/synthetic/synthetic_train.jsonl 2>/dev/null || echo "0 (still starting)" && \
echo "" && \
echo "=== Recent Progress ===" && \
tail -20 synthetic_generation.log 2>/dev/null
```

---

## 7. Estimate completion time
```bash
# Run this after a few minutes to see generation rate
LINES=$(wc -l data/synthetic/synthetic_train.jsonl 2>/dev/null | awk '{print $1}') && \
if [ "$LINES" -gt 0 ]; then \
  PERCENT=$(echo "scale=1; $LINES / 1000" | bc) && \
  echo "Progress: $LINES / 100,000 examples ($PERCENT%)" && \
  echo "Approximately $((100000 - LINES)) examples remaining"; \
else \
  echo "Generation starting..."; \
fi
```

---

## 8. Watch progress automatically (updates every 30 seconds)
```bash
watch -n 30 'wc -l data/synthetic/synthetic_train.jsonl 2>/dev/null || echo "0"'
```
**Exit**: Press Ctrl+C

---

## Quick Check (All-in-One)
```bash
clear && \
echo "🔄 SYNTHETIC DATA GENERATION STATUS" && \
echo "===================================" && \
echo "" && \
ps aux | grep "src.synthesize" | grep -v grep | awk '{print "Process: Running (PID " $2 ", CPU " $3 "%)"}' || echo "Process: Not running" && \
echo "" && \
LINES=$(wc -l data/synthetic/synthetic_train.jsonl 2>/dev/null | awk '{print $1}' || echo "0") && \
PERCENT=$(echo "scale=1; $LINES / 1000" | bc 2>/dev/null || echo "0") && \
echo "Generated: $LINES / 100,000 ($PERCENT%)" && \
echo "" && \
echo "Recent log:" && \
tail -10 synthetic_generation.log 2>/dev/null || echo "  (loading...)"
```

---

## Current Status
As of now, your process is:
- ✅ **Running** (PID 23760, 98% CPU)
- 🔄 **Loading models** (progress bar will appear soon)
- ⏱️ **ETA**: 30-45 minutes from start
