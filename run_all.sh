#!/bin/bash
mkdir -p out/tatqa_logs
for f in examples/tatqa/*.json; do
    base=$(basename "$f" .json)
    python run_amaf.py "$f" > out/tatqa_logs/"${base}_out.txt"
done
