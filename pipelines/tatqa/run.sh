#!/usr/bin/env bash
# AMAF TAT-QA multihop pipeline runner (dataset-specific)
# Logs stored under out/tatqa_logs/<RUN_ID>/

DATASET="tatqa"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/out/${DATASET}_logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "Logs will be stored in $LOG_DIR"

count=0
for f in "$ROOT_DIR"/examples/tatqa/*.json; do
    base=$(basename "$f" .json)
    python "$ROOT_DIR"/run_amaf.py "$f" -d "$DATASET" -p pipelines/tatqa/tatqa_chain.yaml > "$LOG_DIR/${base}_out.txt"
    echo "-------------- Processed $f  ------------------"
    count=$((count+1))
    if [ $count -ge 10 ]; then
        break
    fi
done

echo "Run complete. ${count} files processed. Logs in $LOG_DIR" 