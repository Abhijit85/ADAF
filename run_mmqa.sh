#! /bin/bash
mkdir -p out/mmqa_logs
count=0
for f in examples/mmqa/*.json; do
    base=$(basename "$f" .json)
    python run_amaf.py "$f" -d mmqa > out/mmqa_logs/"${base}_out.txt"
    echo "--------------------------------"
    echo "--------------------------------"
    echo "processing $f"
    count=$((count+1))
    if [ $count -eq 10 ]; then
        break
    fi
done