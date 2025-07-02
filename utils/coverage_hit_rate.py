# utils/coverage_hit_rate.py
from pathlib import Path
import re
import math
import json
from datasets import load_dataset, DownloadConfig

log_dir = Path("out/tatqa_logs")
local_json = Path("data/tatqa/tatqa_dataset_dev.json")  # adjust if elsewhere

# ------------------------------------------------------------------ #
# 1) try HF hub, else local file
# ------------------------------------------------------------------ #
try:
    # timeout=10 s avoids long waits if there's no internet
    ds = load_dataset(
        "next-tat/TAT-QA",
        split="dev",
        download_config=DownloadConfig(timeout=10)
    )
    print("✔ loaded dev split from Hugging Face hub")
except Exception as e:
    print("⚠ could not download from hub:", e)
    print("→ falling back to local JSON:", local_json)
    with local_json.open(encoding="utf-8") as f:
        raw = json.load(f)
    # some dumps wrap in top-level "data"
    ds = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    print(f"✔ loaded {len(ds)} examples locally")

# ------------------------------------------------------------------ #
# fuzzy-numeric matcher (unchanged)
# ------------------------------------------------------------------ #


def normalise(s: str) -> str:
    return re.sub(r"[,$% ]", "", s.lower())


num_re = re.compile(r"-?\d+(?:\.\d+)?")


def numeric_hit(ans: str, summary: str) -> bool:
    if re.search(re.escape(normalise(ans)), normalise(summary)):
        return True
    if num_re.fullmatch(ans.strip()):
        val = float(ans)
        for m in num_re.findall(summary):
            try:
                if math.isclose(float(m), val, rel_tol=0.01):
                    return True
            except ValueError:
                pass
    return False

# ------------------------------------------------------------------ #
# tally coverage
# ------------------------------------------------------------------ #


hits = total = 0
for ex in ds:
    uid = ex.get("id") or ex.get("table", {}).get("uid")
    if not uid:
        continue
    p = log_dir / f"{uid}_out.txt"
    if not p.exists():
        continue

    summary = p.read_text().split("=== FINAL SUMMARY ===")[-1].lower()

    for qa in ex.get("qa_pairs", ex.get("questions", [])):
        ans = str(qa["answer"]).strip()
        if not ans:
            continue
        total += 1
        if numeric_hit(ans, summary):
            hits += 1

print(f"Coverage hit-rate: {hits}/{total}  ->  {hits/total:.2%}")
