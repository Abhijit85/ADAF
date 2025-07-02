#!/usr/bin/env bash
# Run AMAF on FinQA examples into out/finqa_logs/<RUN_ID>/

DATASET="finqa"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="out/${DATASET}_logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "Logs will be stored in $LOG_DIR"

count=0
for f in examples/finqa/*.json; do
    base=$(basename "$f" .json)
    python run_amaf.py "$f" -d "$DATASET" > "$LOG_DIR/${base}_out.txt"
    echo "-------------- Processed $f  ------------------"
    count=$((count+1))
    if [ $count -eq 10 ]; then
        break
    fi
done

echo "Run complete. ${count} files processed. Logs in $LOG_DIR"

python scripts/build_generation_report.py --log_dir "$LOG_DIR" --dataset "$DATASET" || true