# cli/eval_dataset.py
"""Command-line wrapper for evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation import evaluate


def main():
    ap = argparse.ArgumentParser(description="Run ADAF summary/answer evaluation")
    ap.add_argument("--dataset", required=True, choices=["tatqa", "finqa"], help="dataset name")
    ap.add_argument("--gold", required=True, type=Path, help="gold JSON path")
    ap.add_argument("--pred", required=True, type=Path, help="pred directory or JSON file")
    ap.add_argument("--mode", default="summary+echoes", choices=["summary+echoes", "summary"], help="for summary kind predictions")
    ap.add_argument("--pred_kind", default="summary", choices=["summary", "answer"], help="prediction content type")
    ap.add_argument("--out", type=Path, help="output JSON report (optional if --run_id)")
    ap.add_argument("--run_id", type=str, default=None, help="run identifier for results folder")
    ap.add_argument("--eval_id", type=str, default=None, help="evaluation run id (defaults timestamp)")
    ap.add_argument("--openai_key", type=str, help="OpenAI API key for CAE")
    args = ap.parse_args()

    result = evaluate(
        dataset=args.dataset,
        gold_path=args.gold,
        pred_path=args.pred,
        mode=args.mode,
        pred_kind=args.pred_kind,
        openai_key=args.openai_key,
        run_id=args.run_id,
        eval_id=args.eval_id,
    )
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    # console summary
    if "error" in result:
        print(result["error"])
    else:
        print("\n=== Aggregate metrics ===")
        for k, v in result.items():
            if k in ("per_question", "error", "count"):
                continue
            if v is not None:
                print(f"{k:>5}: {v:.4f}")
        print(f"count: {result['count']}")


if __name__ == "__main__":
    main() 