import json
from pathlib import Path
import re

ds_path = Path('data/finqa/finqa_dev.json')
log_dir = Path('out/finqa_logs/')

total = hits = 0
checked = 0

with ds_path.open() as f:
    data = json.load(f)
    for ex in data:
        base = Path(ex['filename']).stem  # e.g., AAL_2013_page_15.pdf
        found = False
        tried = []
        # Try with and without -1 suffix
        for i in range(1, 6):  # Try up to 5 possible questions per file
            log_name = f"{base}-{i}_out.txt"
            tried.append(log_name)
            summ_path = log_dir / log_name
            if summ_path.exists():
                found = True
                break
        if not found:
            # Try without any suffix
            log_name = f"{base}_out.txt"
            tried.append(log_name)
            summ_path = log_dir / log_name
            if summ_path.exists():
                found = True
        if not found:
            # Print the first few misses for debugging
            if checked < 10:
                print(f"No log found for: {base}, tried: {tried}")
                checked += 1
            continue

        summary_text = summ_path.read_text().lower()
        final_summary = ""
        if "=== final summary ===" in summary_text:
            final_summary = summary_text.split("=== final summary ===")[-1].split("===")[0]
        answer_echoes = ""
        if "answer echoes:" in summary_text:
            answer_echoes = summary_text.split("answer echoes:")[-1].split("===")[0]
        combined = final_summary + " " + answer_echoes

        ans = ex.get('qa', {}).get('answer', '')
        ans = str(ans).lower().strip()
        total += 1
        if ans and re.search(re.escape(ans), combined):
            hits += 1

if total == 0:
    print("No answers found in the dev set or no matching logs.")
else:
    print(f"Coverage hit‑rate: {hits}/{total}  ->  {hits/total:.2%}")