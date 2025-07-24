#!/usr/bin/env python3
"""Consolidate predictions of a single AMAF run with dev-set gold answers.

Usage:
    python consolidate_run.py --dataset tatqa --run_dir <path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    # running as module (python -m evaluation.scripts.consolidate_run)
    from evaluation.scripts.datasets import DATASETS  # type: ignore
except ModuleNotFoundError:
    # running as plain script
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.append(str(_P(__file__).resolve().parent))
    from datasets import DATASETS  # type: ignore


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS.keys(), help="Dataset name (tatqa, finqa, fetaqa, mmqa)")
    ap.add_argument("--run_dir", required=True, help="Path to AMAF run directory containing *_out.txt files")
    ap.add_argument("--quiet", action="store_true", help="Suppress warnings")
    args = ap.parse_args(argv)

    if args.quiet:
        import warnings
        warnings.filterwarnings("ignore")

    ds_class = DATASETS[args.dataset]
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