import json
from pathlib import Path
import re

# Path to MMQA dev set and logs
ds_path = Path('data/mmqa/MMQA_dev.jsonl')
log_dir = Path('out/mmqa_logs/')

total = hits = 0

# Read the dev set (jsonl: one json per line)
with ds_path.open() as f:
    for line in f:
        ex = json.loads(line)
        uid = ex.get('qid')
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
                answer_echoes = echoes_parts[1].split("===")[0]
        answer_echoes = answer_echoes.lower()

        # MMQA: usually a single answer per example
        for ans_obj in ex.get('answers', []):
            ans = ans_obj.get('answer', '')
            if isinstance(ans, list):
                ans = ' '.join(str(a) for a in ans)
            ans = str(ans).lower().strip()
            total += 1
            if ans and re.search(re.escape(ans), answer_echoes):
                hits += 1

if total == 0:
    print("No answers found in the dev set or no matching logs.")
else:
    print(f"Coverage hit‑rate: {hits}/{total}  ->  {hits/total:.2%}") 