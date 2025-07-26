#!/usr/bin/env python3
"""Post-process a consolidated JSON file for a dataset.
Adds answer_type, pp_gold_answer, pp_pred_answer for TatQA.

Usage:
    python postprocess_run.py --dataset tatqa --consolidated_file <file>
    python -m evaluation.scripts postprocess --dataset tatqa --consolidated_file <file>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tatqa", "finqa", "fetaqa", "mmqa"])
    ap.add_argument("--consolidated_file", required=True, help="Path to consolidated JSON")
    args = ap.parse_args(argv)

    # allow both script and module usage
    try:
        from evaluation.scripts.postprocess import PROCESSORS  # type: ignore
    except ModuleNotFoundError:
        import sys as _sys, pathlib as _p
        _sys.path.append(str(_p.Path(__file__).resolve().parent))
        from postprocess import PROCESSORS  # type: ignore

    processor_cls = PROCESSORS[args.dataset]
    processor = processor_cls()

    c_path = Path(args.consolidated_file)
    with c_path.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    # pass records through processor (question not needed here)
    enriched = processor.process_records(records)

    out_path = c_path.parent / f"postprocessed_{c_path.name.split('_',1)[1]}"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(enriched, fp, indent=2, ensure_ascii=False)

    # pretty path display (avoid ValueError for mixed abs/rel)
    try:
        display = out_path.relative_to(Path.cwd())
    except ValueError:
        display = out_path
    print(f"Wrote {len(enriched)} records → {display}")


if __name__ == "__main__":
    main() 