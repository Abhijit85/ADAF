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
    outdir="out/finqa_logs/${MODEL_NAME}_${PROVIDER}_${METHOD}_${COMMENT}_$(date +%Y%m%d_%H%M%S)"
else
    outdir="out/finqa_logs/${MODEL_NAME}_${PROVIDER}_${METHOD}_$(date +%Y%m%d_%H%M%S)"
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
for f in examples/finqa/*.json; do
    base=$(basename "$f" .json)
    output_file="$outdir/${base}_out.txt"
    
    echo "Processing $f..."
    if source .venv/bin/activate && python3 run_amaf.py "$f" -d finqa "${FILTERED_ARGS[@]}" > "$output_file" 2>&1; then
        echo "-------------- Processed $f  ------------------"
        ((count++))
    else
        echo "❌ Failed to process $f - removing output file"
        rm -f "$output_file"  # Remove the output file if it was created
    fi
    
    # Add sleep to avoid rate limiting (4 seconds between requests)
    echo "Sleeping for 4 seconds to avoid rate limiting..."
    sleep 4
    
    # if [ $count -ge 10 ]; then
    #     break
    # fi
done

echo "Completed $count examples"

