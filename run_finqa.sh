#! /bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p out/finqa_logs/"$timestamp"
for f in examples/finqa/*.json; do
    base=$(basename "$f" .json)
    python3 run_amaf.py "$f" -d finqa > out/finqa_logs/"$timestamp"/"${base}_out.txt"
    echo "-------------- Processed $f  ------------------"
done