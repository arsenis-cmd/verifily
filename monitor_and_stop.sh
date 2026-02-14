#!/bin/bash
TARGET=20000
PID=28747
LOG="synthetic_generation_cpu.log"

echo "Monitoring synthetic generation (PID: $PID)"
echo "Target: $TARGET examples"
echo "Will stop when reached..."
echo ""

while true; do
  # Get current count
  CURRENT=$(tail -1 "$LOG" | grep -oE "[0-9]+/100000" | head -1 | cut -d'/' -f1)
  
  if [ -n "$CURRENT" ] && [ "$CURRENT" -ge "$TARGET" ]; then
    echo "✅ Target reached! Current: $CURRENT"
    echo "Stopping process $PID..."
    kill -SIGINT $PID
    sleep 5
    
    # Check if output file exists
    if [ -f "data/synthetic/synthetic_train.jsonl" ]; then
      LINES=$(wc -l < data/synthetic/synthetic_train.jsonl)
      echo "Output file has $LINES lines"
      
      # Trim to exactly 20k if needed
      if [ "$LINES" -gt "$TARGET" ]; then
        echo "Trimming to exactly $TARGET lines..."
        head -n $TARGET data/synthetic/synthetic_train.jsonl > data/synthetic/synthetic_train_20k.jsonl
        mv data/synthetic/synthetic_train_20k.jsonl data/synthetic/synthetic_train.jsonl
        echo "✅ Trimmed to $TARGET lines"
      fi
    else
      echo "⚠️ Output file not found!"
    fi
    
    break
  fi
  
  echo "[$(date +%H:%M:%S)] Progress: $CURRENT/$TARGET ($(($TARGET - $CURRENT)) remaining)"
  sleep 60
done

echo ""
echo "✅ Synthetic generation stopped at 20k examples!"
echo "Ready to train Model C"
