#!/bin/bash
mkdir -p out/tatqa_logs/15_06_2025
count=0
for f in examples/tatqa/*.json; do
    base=$(basename "$f" .json)
    python run_amaf.py "$f" > out/tatqa_logs/15_06_2025/"${base}_out.txt"
    echo "Processed $base"
    count=$((count + 1))
    if [ $count -gt 30 ]; then
        break
    fi
done
