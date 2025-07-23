#!/bin/bash

# keep a pristine copy of the CLI arguments
ORIG_ARGS=("$@")

# ── derive MODEL_NAME for folder naming ───────────────────────────
MODEL_NAME="defaultmodel"
PROVIDER="defaultprovider"
COMMENT=""
METHOD=""

# scan the copy ONLY to pick out --model, --provider, --comment, and --method
for (( i=0; i<${#ORIG_ARGS[@]}; i++ )); do
    case "${ORIG_ARGS[$i]}" in
        --model)
            MODEL_NAME="${ORIG_ARGS[$((i+1))]}"
            ;;
        --model=*)
            MODEL_NAME="${ORIG_ARGS[$i]#--model=}"
            ;;
        --provider)
            PROVIDER="${ORIG_ARGS[$((i+1))]}"
            ;;
        --provider=*)
            PROVIDER="${ORIG_ARGS[$i]#--provider=}"
            ;;
        --comment)
            COMMENT="${ORIG_ARGS[$((i+1))]}"
            ;;
        --comment=*)
            COMMENT="${ORIG_ARGS[$i]#--comment=}"
            ;;
        --method)
            METHOD="${ORIG_ARGS[$((i+1))]}"
            ;;
        --method=*)
            METHOD="${ORIG_ARGS[$i]#--method=}"
            ;;
    esac
done

echo "Using model: $MODEL_NAME"
echo "Using provider: $PROVIDER"
if [ -n "$METHOD" ]; then
    echo "Using method: $METHOD"
fi
if [ -n "$COMMENT" ]; then
    echo "Using comment: $COMMENT"
fi

if [ -n "$COMMENT" ]; then
    outdir="out/fetaqa_logs/${MODEL_NAME}_${PROVIDER}_${METHOD}_${COMMENT}_$(date +%Y%m%d_%H%M%S)"
else
    outdir="out/fetaqa_logs/${MODEL_NAME}_${PROVIDER}_${METHOD}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$outdir"

# Build a new array without --comment and its value
FILTERED_ARGS=()
skip_next=0
for (( i=0; i<${#ORIG_ARGS[@]}; i++ )); do
    if [ $skip_next -eq 1 ]; then
        skip_next=0
        continue
    fi
    case "${ORIG_ARGS[$i]}" in
        --comment)
            skip_next=1
            ;;
        --comment=*)
            ;;
        *)
            FILTERED_ARGS+=("${ORIG_ARGS[$i]}")
            ;;
    esac
done

count=0
for f in examples/fetaqa/*.json; do
    base=$(basename "$f" .json)
    python3 run_amaf.py "$f" -d fetaqa "${FILTERED_ARGS[@]}" > "$outdir/${base}_out.txt"
    echo "--------------------------------"
    echo "processing $f"
    ((count++))
    if [ $count -ge 10 ]; then
        break
    fi
done

echo "Completed $count examples" 