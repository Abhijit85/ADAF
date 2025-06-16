#!/bin/bash
mkdir -p out/finqa_logs
count=0
for f in examples/finqa/*.json; do
    base=$(basename "$f" .json)
    python run_amaf.py "$f" > out/finqa_logs/"${base}_out.txt"
    echo "Processed $base"
    count=$((count + 1))
    if [ $count -gt 10 ]; then
        break
    fi
done 