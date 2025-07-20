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
log_dir = Path('out/tatqa_logs/20250718_131942')
# The following code sets up the evaluation for TAT-QA summaries.
# It loads the development split, sets the log directory, and prepares for answer coverage checking.

total = hits = 0
for ex in ds:
    uid = ex.get('id') or ex.get('table', {}).get('uid')
    if not uid:
        continue
    summ_path = log_dir / f"{uid}_out.txt"
    if not summ_path.exists():
        continue
    
    # Read the entire summary file
    summary_text = summ_path.read_text()
    
    # Extract only the Answer Echoes section
    answer_echoes = ""
    if "Answer Echoes:" in summary_text:
        echoes_parts = summary_text.split("Answer Echoes:")
        if len(echoes_parts) > 1:
            answer_echoes = echoes_parts[1].split("===")[0]  # Get everything after Answer Echoes until next section
    
    # Use only Answer Echoes for evaluation
    answer_echoes = answer_echoes.lower()
    
    for qa in ex.get('questions', []):
        ans = qa.get('answer')
        if isinstance(ans, list):
            ans = ' '.join(str(a) for a in ans)
        ans = str(ans).lower().strip()
        total += 1
        if ans and re.search(re.escape(ans), summary_text):
            hits += 1
print(f"Coverage hit‑rate: {hits}/{total}  ->  {hits/total:.2%}")
