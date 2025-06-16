# Note: This script assumes that the summaries are stored in a specific format.
# Adjust the regex and logic as needed based on your actual summary format.
# The script reads the dataset, checks each summary for the presence of the answer,
# and counts how many answers are covered by the summaries.
# The output is the hit-rate of answers covered by the summaries.
# This script is useful for evaluating the performance of a model on the TAT-QA dataset
# in terms of how well it captures the answers in its summaries.

import json
from pathlib import Path
import re

# Load the local TAT-QA development split
ds_path = Path('data/TATQA/tatqa_dataset_dev.json')

with ds_path.open() as f:
    ds = json.load(f)
log_dir = Path('out/tatqa_logs/15_06_2025')

total = hits = 0
count = 0
for ex in ds:
    uid = ex.get('id') or ex.get('table', {}).get('uid')
    if not uid:
        continue
    summ_path = log_dir / f"{uid}_out.txt"
    if not summ_path.exists():
        continue
    summary = summ_path.read_text().split('=== FINAL SUMMARY ===')[-1].lower()
    for qa in ex.get('questions', []):
        ans = qa.get('answer')
        if isinstance(ans, list):
            ans = ' '.join(str(a) for a in ans)
        ans = str(ans).lower().strip()
        total += 1
        if ans and re.search(re.escape(ans), summary):
            hits += 1
    count += 1
    if count > 129:
        break
print(f"Coverage hit‑rate: {hits}/{total}  ->  {hits/total:.2%}")
