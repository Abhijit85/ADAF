#!/usr/bin/env python3
"""Score a post-processed JSON file and output per-question and averaged metrics.

Usage:
    python score_run.py --post_file evaluation/tatqa/<run>/postprocessed_<ts>.json
    python -m evaluation.scripts score --dataset tatqa --post_file evaluation/tatqa/<run>/postprocessed_<ts>.json
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

    try:
        from evaluation.scripts.datasets.tatqa import TatqaDataset  # type: ignore
    except ModuleNotFoundError:
        from datasets.tatqa import TatqaDataset  # type: ignore

    NULL_TOKENS = {"", "null", "none"}

    logging.basicConfig(level=logging.INFO)

    def _is_null_pred(pred: str) -> bool:
        p = pred.lower().strip()
        if p in NULL_TOKENS:
            return True
        if "cannot answer" in p:
            return True
        if "insufficient" in p:
            return True
        return False

    post_path = Path(args.post_file)
    gem_key = args.gemini_key or os.getenv("GEMINI_API_KEY")
    with post_path.open("r", encoding="utf-8") as fp:
        records: List[Dict[str, Any]] = json.load(fp)

    scored: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Build lookup map: qid -> question text (only for TatQA currently)
    # ------------------------------------------------------------------
    dataset_name = post_path.parents[1].name  # evaluation/<dataset>/..
    qmap: Dict[str, str] = {}
    if dataset_name == "tatqa" and hasattr(TatqaDataset, "load_gold_source"):
        ds = TatqaDataset()
        try:
            gold_src = ds.load_gold_source()
            qmap = {str(q["uid"]): q.get("question", "") for item in gold_src for q in item.get("questions", [])}
        except Exception as exc:
            logging.warning(f"Failed to build TatQA question map: {exc}")

    # accumulators
    counts = defaultdict(int)
    sum_exact = defaultdict(float)
    sum_f1 = defaultdict(float)
    sum_cae = defaultdict(float)
    sum_hcs = defaultdict(float)
    null_counts = defaultdict(int)
    non_null_counts = defaultdict(int)
    sum_exact_non = defaultdict(float)
    sum_f1_non = defaultdict(float)
    sum_cae_non = defaultdict(float)
    sum_hcs_non = defaultdict(float)
    # separate denominators for CAE/HCS because some questions may be skipped
    cnt_cae = defaultdict(int)
    cnt_hcs = defaultdict(int)

    for rec in records:
        atype = rec.get("answer_type")
        if not atype:
            continue
        g = str(rec.get("pp_gold_answer", ""))
        if not g.strip():
            continue
        p = str(rec.get("pp_pred_answer", ""))

        # FinQA-specific scoring
        if dataset_name == "finqa":
            # Use the is_close field from postprocessing
            is_close = rec.get("is_close", False)
            cannot_determine = rec.get("cannot_determine", False)
            
            # F1: Full credit for honest assessment
            # EM: Partial credit for honest assessment
            if is_close and cannot_determine:
                em = 0.5  # Partial credit for honest assessment
                f1 = 1.0  # Full credit for honest assessment
            elif is_close:
                em = 1
                f1 = 1.0
            else:
                em = 0
                f1 = 0.0
            
            rec["exact_match"] = em
            rec["f1"] = f1
            
            # Group by source for FinQA
            source = rec.get("source", "unknown")
            if source:
                # Different scoring strategies based on source
                if source == "table":
                    # Table source: focus on exact numeric accuracy
                    pass
                elif source == "text":
                    # Text source: focus on narrative accuracy
                    pass
                elif source == "combined":
                    # Combined source: balanced approach
                    pass
        else:
            # Standard scoring for other datasets
            em = exact_match(g, p)
            f1 = f1_score(g, p)
            rec["exact_match"] = em
            rec["f1"] = f1
        
        # Source-aware scoring for FETA-QA
        source = rec.get("source", "")
        if source and dataset_name == "fetaqa":
            # Different scoring strategies based on source
            if source == "data":
                # Data source: focus on numerical accuracy
                # Could add additional numerical validation here
                pass
            elif source == "text":
                # Text source: focus on narrative accuracy
                # Could add additional narrative validation here
                pass
            elif source == "combined":
                # Combined source: balanced approach
                # Could add additional combined validation here
                pass
        
        # --------------------------------------------------------------
        # CAE + HCS
        # --------------------------------------------------------------

        question_text = qmap.get(rec.get("qid", rec.get("feta_id", "")), "")

        if not question_text:
            logging.warning(f"Question text missing for qid {rec.get('qid', rec.get('feta_id', ''))}; skipping CAE/HCS computation.")
            cae_val = None  # mark as not computed
            hcs_val = None
        else:
            gold_for_cae = g  # use post-processed gold answer directly
            pred_for_cae = p  # use post-processed pred answer directly
            cae_val = cae_score(question=question_text, gold=gold_for_cae, pred=pred_for_cae, gemini_key=gem_key)
            hcs_val = 1 if (f1 >= 0.8 and cae_val == 1.0) else 0

        rec["cae"] = cae_val if cae_val is not None else "NA"
        rec["hcs"] = hcs_val if hcs_val is not None else "NA"

        counts["overall"] += 1
        sum_exact["overall"] += em
        sum_f1["overall"] += f1
        # aggregate metrics only if computed
        if cae_val is not None:
            sum_cae["overall"] += cae_val
            sum_hcs["overall"] += hcs_val
            cnt_cae["overall"] += 1
            cnt_hcs["overall"] += 1

        # Group by answer_type for most datasets, but by source for FinQA
        if dataset_name == "finqa":
            group_key = rec.get("source", "unknown") or "unknown"
        else:
            group_key = atype
            
        counts[group_key] += 1
        sum_exact[group_key] += em
        sum_f1[group_key] += f1
        if cae_val is not None:
            sum_cae[group_key] += cae_val
            sum_hcs[group_key] += hcs_val
            cnt_cae[group_key] += 1
            cnt_hcs[group_key] += 1

        is_null = _is_null_pred(p)
        if is_null:
            null_counts["overall"] += 1
            null_counts[group_key] += 1
        else:
            non_null_counts["overall"] += 1
            non_null_counts[group_key] += 1
            sum_exact_non["overall"] += em
            sum_f1_non["overall"] += f1
            sum_exact_non[group_key] += em
            sum_f1_non[group_key] += f1
            if cae_val is not None:
                sum_cae_non["overall"] += cae_val
                sum_hcs_non["overall"] += hcs_val
            if cae_val is not None:
                sum_cae_non[group_key] += cae_val
                sum_hcs_non[group_key] += hcs_val

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
            "cae": round(sum_cae[key]/cnt_cae[key],4) if cnt_cae.get(key) else None,
            "hcs": round(sum_hcs[key]/cnt_hcs[key],4) if cnt_hcs.get(key) else None,
            "null_or_na": null_counts.get(key, 0),
        }
        # metrics excluding nulls
        if non_null_counts.get(key):
            entry["exact_no_null"] = round(sum_exact_non[key]/non_null_counts[key],4)
            entry["f1_no_null"] = round(sum_f1_non[key]/non_null_counts[key],4)
            entry["cae_no_null"] = round(sum_cae_non[key]/cnt_cae.get(key, non_null_counts[key]),4) if cnt_cae.get(key) else None
            entry["hcs_no_null"] = round(sum_hcs_non[key]/cnt_hcs.get(key, non_null_counts[key]),4) if cnt_hcs.get(key) else None
            entry["count_no_null"] = non_null_counts[key]
        else:
            entry["exact_no_null"] = None
            entry["f1_no_null"] = None
            entry["cae_no_null"] = None
            entry["hcs_no_null"] = None
            entry["count_no_null"] = 0
        if key == "overall":
            avg_out["overall"] = entry
        else:
            # Use appropriate grouping name based on dataset
            if dataset_name == "finqa":
                avg_out.setdefault("by_source", {})[key] = entry
            else:
                avg_out.setdefault("by_answer_type", {})[key] = entry

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