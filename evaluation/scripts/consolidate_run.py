#!/usr/bin/env python3
"""Consolidate predictions of a single AMAF run with dev-set gold answers.

Usage:
    python consolidate_run.py --dataset tatqa --run_dir <path>
    python -m evaluation.scripts consolidate --dataset tatqa --run_dir <path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tatqa", "finqa", "fetaqa", "fetaqa_perturbed", "mmqa"], help="Dataset name")
    ap.add_argument("--run_dir", required=True, help="Path to AMAF run directory containing *_out.txt files")
    ap.add_argument("--quiet", action="store_true", help="Suppress warnings")
    ap.add_argument("--perturbation_class", help="Perturbation class for fetaqa_perturbed dataset")
    ap.add_argument("--perturbation_type", help="Perturbation type for fetaqa_perturbed dataset")
    args = ap.parse_args(argv)

    if args.quiet:
        import warnings
        warnings.filterwarnings("ignore")

    # Handle imports with fallback
    try:
        # running as module (python -m evaluation.scripts.consolidate_run)
        from evaluation.scripts.datasets import DATASETS  # type: ignore
    except ModuleNotFoundError:
        # running as plain script
        from pathlib import Path as _P
        import sys as _sys
        _sys.path.append(str(_P(__file__).resolve().parent))
        from datasets import DATASETS  # type: ignore

    ds_class = DATASETS[args.dataset]
    
    # Pass perturbation parameters for fetaqa_perturbed dataset
    if args.dataset == "fetaqa_perturbed":
        ds = ds_class(perturbation_class=args.perturbation_class, perturbation_type=args.perturbation_type)
    else:
        ds = ds_class()

    print("[CONSOLIDATE] Loading gold answers …", file=sys.stderr)
    gold = ds.load_gold()
    print(f"[CONSOLIDATE] Loaded {len(gold):,} gold QA pairs", file=sys.stderr)

    print("[CONSOLIDATE] Parsing predictions …", file=sys.stderr)
    preds = ds.parse_run_dir(args.run_dir)
    print(f"[CONSOLIDATE] Parsed {len(preds):,} predictions", file=sys.stderr)

    print("[CONSOLIDATE] Building unified JSON …", file=sys.stderr)
    records = ds.consolidate(gold, preds)
    out_path = ds.dump(records, args.dataset, args.run_dir)

    print(f"[DONE] Wrote {len(records):,} records → {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main() 