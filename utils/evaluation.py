# Note: This script assumes that the summaries are stored in a specific format.
# Adjust the regex and logic as needed based on your actual summary format.
# The script reads the dataset, checks each summary for the presence of the answer,
# and counts how many answers are covered by the summaries.
# The output is the hit-rate of answers covered by the summaries.
# This script is useful for evaluating the performance of a model on the TAT-QA dataset
# in terms of how well it captures the answers in its summaries.

from datasets import load_dataset
from pathlib import Path
import re

ds = load_dataset('next-tat/TAT-QA', 'v1.1', split='dev')
log_dir = Path('out/tatqa_logs')

total = hits = 0
for ex in ds:
    summ_path = log_dir / f"{ex['id']}_out.txt"
    if not summ_path.exists():
        continue
    summary = summ_path.read_text().split('=== FINAL SUMMARY ===')[-1].lower()
    for qa in ex['qa_pairs']:
        ans = str(qa['answer']).lower().strip()
        total += 1
        if ans and re.search(re.escape(ans), summary):
            hits += 1
print(f"Coverage hit‑rate: {hits}/{total}  ->  {hits/total:.2%}")
