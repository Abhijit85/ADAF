#! /bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p out/mmqa_logs/"$timestamp"
for f in examples/mmqa/*.json; do
    base=$(basename "$f" .json)
    python3 run_amaf.py "$f" -d mmqa > out/mmqa_logs/"$timestamp"/"${base}_out.txt"
    echo "--------------------------------"
    echo "--------------------------------"
    echo "processing $f"
done