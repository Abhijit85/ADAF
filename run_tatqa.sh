#!/bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p out/tatqa_logs/$timestamp/
count=0
for f in examples/tatqa/*.json; do
    base=$(basename "$f" .json)
    python3 run_amaf.py "$f" -d tatqa > out/tatqa_logs/$timestamp/"${base}_out.txt"
    echo "-------------- Processed $f  ------------------"
    if [ $count -eq 10 ]; then
        break
    fi
    count=$((count+1))
done
