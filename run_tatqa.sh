#!/bin/bash
mkdir -p out/tatqa_logs/280/
count=0
for f in examples/tatqa/*.json; do
    base=$(basename "$f" .json)
    python run_amaf.py "$f" -d tatqa > out/tatqa_logs/280/"${base}_out.txt"
    echo "-------------- Processed $f  ------------------"
    if [ $count -eq 10 ]; then
        break
    fi
    count=$((count+1))
done
