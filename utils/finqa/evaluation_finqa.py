import json
from pathlib import Path
import re

# Load the FinQA development split
ds_path = Path('data/Finqa /finqa_dev.json')  # Note the space after Finqa
with ds_path.open() as f:
    ds = json.load(f)

log_dir = Path('out/finqa_logs')

total = hits = 0
count = 0
for ex in ds:
    uid = ex.get('id', '')
    if not uid:
        continue
    
    # Convert uid to match the output filename format
    safe_uid = f"finqa_{uid.replace('/', '_')}"
    summ_path = log_dir / f"{safe_uid}_out.txt"
    
    if not summ_path.exists():
        continue
        
    summary = summ_path.read_text().split('=== FINAL SUMMARY ===')[-1].lower()
    
    # Get the answer from the qa dictionary
    answer = ex.get('qa', {}).get('answer', '')
    print(answer)
    if isinstance(answer, list):
        answer = ' '.join(str(a) for a in answer)
    answer = str(answer).lower().strip()
    
    total += 1
    if answer and re.search(re.escape(answer), summary):
        hits += 1
        print(summary)
        print(answer)
        print('--------------------------------')
    count += 1
    if count > 10:  # Limit to first 10 examples for testing
        break

print(f"Coverage hit‑rate: {hits}/{total}  ->  {hits/total:.2%}")