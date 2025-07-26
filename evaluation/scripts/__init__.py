#!/usr/bin/env python3
"""CLI for evaluation scripts.

Usage:
    python -m evaluation.scripts consolidate --dataset fetaqa --run_dir <path>
    python -m evaluation.scripts postprocess --dataset fetaqa --consolidated_file <file>
    python -m evaluation.scripts score --dataset fetaqa --post_file <file>
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluation pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Consolidate command
    consolidate_parser = subparsers.add_parser("consolidate", help="Consolidate predictions with gold answers")
    consolidate_parser.add_argument("--dataset", required=True, choices=["tatqa", "finqa", "fetaqa", "mmqa"])
    consolidate_parser.add_argument("--run_dir", required=True, help="Path to run directory")
    consolidate_parser.add_argument("--quiet", action="store_true", help="Suppress warnings")
    
    # Postprocess command
    postprocess_parser = subparsers.add_parser("postprocess", help="Post-process consolidated predictions")
    postprocess_parser.add_argument("--dataset", required=True, choices=["tatqa", "finqa", "fetaqa", "mmqa"])
    postprocess_parser.add_argument("--consolidated_file", required=True, help="Path to consolidated JSON")
    
    # Score command
    score_parser = subparsers.add_parser("score", help="Score post-processed predictions")
    score_parser.add_argument("--dataset", required=True, choices=["tatqa", "finqa", "fetaqa", "mmqa"])
    score_parser.add_argument("--post_file", required=True, help="Path to post-processed JSON")
    score_parser.add_argument("--gemini_key", help="Optional Gemini API key")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate command
    if args.command == "consolidate":
        from .consolidate_run import main as consolidate_main
        consolidate_main(["--dataset", args.dataset, "--run_dir", args.run_dir] + (["--quiet"] if args.quiet else []))
    
    elif args.command == "postprocess":
        from .postprocess_run import main as postprocess_main
        postprocess_main(["--dataset", args.dataset, "--consolidated_file", args.consolidated_file])
    
    elif args.command == "score":
        if args.dataset == "fetaqa":
            from .score_run_fetaqa_enhanced import main as score_main
            score_args = ["--post_file", args.post_file]
            if args.gemini_key:
                score_args.extend(["--gemini_key", args.gemini_key])
            score_main(score_args)
        else:
            from .score_run import main as score_main
            score_args = ["--post_file", args.post_file]
            if args.gemini_key:
                score_args.extend(["--gemini_key", args.gemini_key])
            score_main(score_args)

if __name__ == "__main__":
    main() 