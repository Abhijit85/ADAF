#! /bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p out/fetaqa_logs/"$timestamp"
count=0
for f in examples/fetaqa/*.json; do
    base=$(basename "$f" .json)
    python3 run_amaf.py "$f" -d fetaqa > out/fetaqa_logs/"$timestamp"/"${base}_out.txt"
    echo "--------------------------------"
    echo "--------------------------------"
    echo "processing $f"
    count=$((count+1))
    if [ $count -eq 10 ]; then
        break
    fi
done 