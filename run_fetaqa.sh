#! /bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p out/fetaqa_logs/"$timestamp"
for f in examples/fetaqa/*.json; do
    base=$(basename "$f" .json)
    python3 run_amaf.py "$f" -d fetaqa > out/fetaqa_logs/"$timestamp"/"${base}_out.txt"
    echo "--------------------------------"
    echo "--------------------------------"
    echo "processing $f"
done 