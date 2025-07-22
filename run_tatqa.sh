#!/bin/bash

# keep a pristine copy of the CLI arguments
ORIG_ARGS=("$@")

# ── derive MODEL_NAME for folder naming ───────────────────────────
MODEL_NAME="defaultmodel"

# scan the copy ONLY to pick out --model
for (( i=0; i<${#ORIG_ARGS[@]}; i++ )); do
    case "${ORIG_ARGS[$i]}" in
        --model)
            MODEL_NAME="${ORIG_ARGS[$((i+1))]}"
            ;;
        --model=*)
            MODEL_NAME="${ORIG_ARGS[$i]#--model=}"
            ;;
    esac
done

echo "Using model: $MODEL_NAME"

timestamp=$(date +%Y%m%d_%H%M%S)
outdir="out/tatqa_logs/${MODEL_NAME}_${timestamp}"
mkdir -p "$outdir"

count=0
for f in examples/tatqa/*.json; do
    base=$(basename "$f" .json)
    python3 run_amaf.py "$f" -d tatqa "${ORIG_ARGS[@]}" > "$outdir/${base}_out.txt"
    echo "-------------- Processed $f  ------------------"
    ((count++))
    if [ $count -ge 10 ]; then
        break
    fi
done

echo "Completed $count examples"
