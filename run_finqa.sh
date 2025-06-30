#! /bin/bash
mkdir -p out/finqa_logs
count=0
for f in examples/finqa/*.json; do
    base=$(basename "$f" .json)
    python run_amaf.py "$f" -d finqa > out/finqa_logs/"${base}_out.txt"
    count=$((count+1))
    if [ $count -eq 10 ]; then
        break
    fi
    echo "-------------- Processed $f  ------------------"
done