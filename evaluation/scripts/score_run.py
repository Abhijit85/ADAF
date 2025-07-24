#!/usr/bin/env python3
"""Score a post-processed JSON file and output per-question and averaged metrics.

Usage:
    python score_run.py --post_file evaluation/tatqa/<run>/postprocessed_<ts>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

try:
    from evaluation.scripts.scoring.basic import exact_match, f1_score  # type: ignore
except ModuleNotFoundError:
    from scoring.basic import exact_match, f1_score  # type: ignore


NULL_TOKENS = {"", "null", "none"}


def _is_null_pred(pred: str) -> bool:
    p = pred.lower().strip()
    if p in NULL_TOKENS:
        return True
    if "cannot answer" in p:
        return True
    if "insufficient" in p:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post_file", required=True, help="Path to post-processed JSON")
    args = ap.parse_args()

    post_path = Path(args.post_file)
    with post_path.open("r", encoding="utf-8") as fp:
        records: List[Dict[str, Any]] = json.load(fp)

    scored: List[Dict[str, Any]] = []

    # accumulators
    counts = defaultdict(int)
    sum_exact = defaultdict(float)
    sum_f1 = defaultdict(float)
    null_counts = defaultdict(int)
    non_null_counts = defaultdict(int)
    sum_exact_non = defaultdict(float)
    sum_f1_non = defaultdict(float)

    for rec in records:
        atype = rec.get("answer_type")
        if not atype:
            continue
        g = str(rec.get("pp_gold_answer", ""))
        if not g.strip():
            continue
        p = str(rec.get("pp_pred_answer", ""))

        em = exact_match(g, p)
        f1 = f1_score(g, p)
        rec["exact_match"] = em
        rec["f1"] = f1
        rec["cae"] = None  # placeholder
        rec["hcs"] = None  # placeholder

        counts["overall"] += 1
        sum_exact["overall"] += em
        sum_f1["overall"] += f1

        counts[atype] += 1
        sum_exact[atype] += em
        sum_f1[atype] += f1

        is_null = _is_null_pred(p)
        if is_null:
            null_counts["overall"] += 1
            null_counts[atype] += 1
        else:
            non_null_counts["overall"] += 1
            non_null_counts[atype] += 1
            sum_exact_non["overall"] += em
            sum_f1_non["overall"] += f1
            sum_exact_non[atype] += em
            sum_f1_non[atype] += f1

        scored.append(rec)

    # ----- output files -----
    ts_part = post_path.name.split("_", 1)[1]  # keep timestamp suffix
    scored_path = post_path.parent / f"scored_{ts_part}"
    avg_path = post_path.parent / f"avg_scores_{ts_part}"

    with scored_path.open("w", encoding="utf-8") as fp:
        json.dump(scored, fp, indent=2, ensure_ascii=False)

    avg_out = {"overall": {}, "by_answer_type": {}}
    for key in counts:
        avg_exact = sum_exact[key] / counts[key] if counts[key] else 0.0
        avg_f1 = sum_f1[key] / counts[key] if counts[key] else 0.0
        # overall metrics incl. nulls
        entry = {
            "count": counts[key],
            "exact": round(avg_exact, 4),
            "f1": round(avg_f1, 4),
            "null_or_na": null_counts.get(key, 0),
        }
        # metrics excluding nulls
        if non_null_counts.get(key):
            entry["exact_no_null"] = round(sum_exact_non[key]/non_null_counts[key],4)
            entry["f1_no_null"] = round(sum_f1_non[key]/non_null_counts[key],4)
            entry["count_no_null"] = non_null_counts[key]
        else:
            entry["exact_no_null"] = None
            entry["f1_no_null"] = None
            entry["count_no_null"] = 0
        if key == "overall":
            avg_out["overall"] = entry
        else:
            avg_out["by_answer_type"][key] = entry

    with avg_path.open("w", encoding="utf-8") as fp:
        json.dump(avg_out, fp, indent=2, ensure_ascii=False)

    for pth, label in ((scored_path, "per-question scores"), (avg_path, "averaged scores")):
        try:
            disp = pth.relative_to(Path.cwd())
        except ValueError:
            disp = pth
        print(f"Wrote {label} → {disp}")


if __name__ == "__main__":
    main() 