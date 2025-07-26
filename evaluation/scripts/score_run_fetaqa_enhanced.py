#!/usr/bin/env python3
"""FETA-QA scorer with enhanced post-processing and entity matching.

Usage:
    python score_run_fetaqa_enhanced.py --post_file evaluation/fetaqa/<run>/postprocessed_<ts>.json
    python -m evaluation.scripts score --dataset fetaqa --post_file evaluation/fetaqa/<run>/postprocessed_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import os
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--post_file", required=True, help="Path to post-processed JSON")
    ap.add_argument("--gemini_key", help="Optional Gemini API key; defaults to GEMINI_API_KEY env var")
    args = ap.parse_args(argv)

    # Handle imports with fallback
    try:
        from evaluation.scripts.scoring.basic import exact_match, f1_score  # type: ignore
    except ModuleNotFoundError:
        from scoring.basic import exact_match, f1_score  # type: ignore

    try:
        from evaluation.scripts.scoring.cae import cae_score  # type: ignore
    except ModuleNotFoundError:
        from scoring.cae import cae_score  # type: ignore

    logging.basicConfig(level=logging.INFO)

    NULL_TOKENS = {"", "null", "none"}

    def _is_null_pred(p: str) -> bool:
        p = p.lower().strip()
        if p in NULL_TOKENS:
            return True
        if "cannot answer" in p:
            return True
        if "insufficient" in p:
            return True
        return False

    def _calculate_entity_score(gold_entities: List[str], pred_entities: List[str]) -> float:
        """Calculate entity matching score."""
        if not gold_entities:
            return 0.0
        
        gold_set = set(gold_entities)
        pred_set = set(pred_entities)
        
        common = len(gold_set & pred_set)
        return common / len(gold_set)

    post_path = Path(args.post_file)
    gem_key = args.gemini_key or os.getenv("GEMINI_API_KEY")
    with post_path.open("r", encoding="utf-8") as fp:
        records: List[Dict[str, Any]] = json.load(fp)

    scored: List[Dict[str, Any]] = []

    # Aggregators keyed by source value
    counts = defaultdict(int)
    sum_exact = defaultdict(float)
    sum_f1 = defaultdict(float)
    sum_cae = defaultdict(float)
    sum_hcs = defaultdict(float)
    sum_entity = defaultdict(float)  # New entity score
    null_counts = defaultdict(int)
    non_null_counts = defaultdict(int)
    sum_exact_non = defaultdict(float)
    sum_f1_non = defaultdict(float)
    sum_cae_non = defaultdict(float)
    sum_hcs_non = defaultdict(float)
    sum_entity_non = defaultdict(float)
    cnt_cae = defaultdict(int)
    cnt_hcs = defaultdict(int)

    for rec in records:
        src_key = rec.get("source", "unknown") or "unknown"

        g = str(rec.get("pp_gold_answer", ""))
        p = str(rec.get("pp_pred_answer", ""))
        if not g.strip():
            continue  # skip if no gold

        em = exact_match(g, p)
        f1 = f1_score(g, p)
        
        # Calculate entity score
        gold_entities = rec.get("gold_entities", [])
        pred_entities = rec.get("pred_entities", [])
        entity_score = _calculate_entity_score(gold_entities, pred_entities)
        
        rec["exact_match"] = em
        rec["f1"] = f1
        rec["entity_score"] = entity_score

        qtext = rec.get("question", "")
        cae_val = cae_score(question=qtext, gold=g, pred=p, gemini_key=gem_key)
        hcs_val = 1 if (f1 >= 0.8 and cae_val == 1.0) else 0
        rec["cae"] = cae_val
        rec["hcs"] = hcs_val

        # overall
        counts["overall"] += 1
        sum_exact["overall"] += em
        sum_f1["overall"] += f1
        sum_entity["overall"] += entity_score
        sum_cae["overall"] += cae_val
        sum_hcs["overall"] += hcs_val
        cnt_cae["overall"] += 1
        cnt_hcs["overall"] += 1

        # per-source
        counts[src_key] += 1
        sum_exact[src_key] += em
        sum_f1[src_key] += f1
        sum_entity[src_key] += entity_score
        sum_cae[src_key] += cae_val
        sum_hcs[src_key] += hcs_val
        cnt_cae[src_key] += 1
        cnt_hcs[src_key] += 1

        is_null = _is_null_pred(p)
        if is_null:
            null_counts["overall"] += 1
            null_counts[src_key] += 1
        else:
            non_null_counts["overall"] += 1
            non_null_counts[src_key] += 1
            sum_exact_non["overall"] += em
            sum_f1_non["overall"] += f1
            sum_entity_non["overall"] += entity_score
            sum_cae_non["overall"] += cae_val
            sum_hcs_non["overall"] += hcs_val
            sum_exact_non[src_key] += em
            sum_f1_non[src_key] += f1
            sum_entity_non[src_key] += entity_score
            sum_cae_non[src_key] += cae_val
            sum_hcs_non[src_key] += hcs_val

        scored.append(rec)

    # write outputs
    ts_part = post_path.name.split("_", 1)[1]  # keep timestamp suffix
    out_dir = post_path.parent
    scored_path = out_dir / f"scored_{ts_part}"
    avg_path = out_dir / f"avg_scores_{ts_part}"

    with scored_path.open("w", encoding="utf-8") as fp:
        json.dump(scored, fp, indent=2, ensure_ascii=False)

    avg_out = {"overall": {}, "by_source": {}}
    for key in counts:
        avg_exact = sum_exact[key] / counts[key] if counts[key] else 0.0
        avg_f1 = sum_f1[key] / counts[key] if counts[key] else 0.0
        avg_entity = sum_entity[key] / counts[key] if counts[key] else 0.0
        avg_cae = sum_cae[key] / cnt_cae[key] if cnt_cae[key] else None
        avg_hcs = sum_hcs[key] / cnt_hcs[key] if cnt_hcs[key] else None

        entry = {
            "count": counts[key],
            "exact": round(avg_exact, 4),
            "f1": round(avg_f1, 4),
            "entity_score": round(avg_entity, 4),
            "cae": round(avg_cae, 4) if avg_cae is not None else None,
            "hcs": round(avg_hcs, 4) if avg_hcs is not None else None,
            "null_or_na": null_counts.get(key, 0),
        }
        # no-null metrics
        if non_null_counts.get(key):
            entry["exact_no_null"] = round(sum_exact_non[key]/non_null_counts[key], 4)
            entry["f1_no_null"] = round(sum_f1_non[key]/non_null_counts[key], 4)
            entry["entity_score_no_null"] = round(sum_entity_non[key]/non_null_counts[key], 4)
            entry["cae_no_null"] = round(sum_cae_non[key]/non_null_counts[key], 4)
            entry["hcs_no_null"] = round(sum_hcs_non[key]/non_null_counts[key], 4)
            entry["count_no_null"] = non_null_counts[key]
        else:
            entry["exact_no_null"] = None
            entry["f1_no_null"] = None
            entry["entity_score_no_null"] = None
            entry["cae_no_null"] = None
            entry["hcs_no_null"] = None
            entry["count_no_null"] = 0

        if key == "overall":
            avg_out["overall"] = entry
        else:
            avg_out["by_source"][key] = entry

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