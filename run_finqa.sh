#! /bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p out/finqa_logs/"$timestamp"
count=0
for f in examples/finqa/*.json; do
    base=$(basename "$f" .json)
    python3 run_amaf.py "$f" -d finqa > out/finqa_logs/"$timestamp"/"${base}_out.txt"
    echo "-------------- Processed $f  ------------------"
    count=$((count+1))
    if [ $count -eq 10 ]; then
        break
    fi
done