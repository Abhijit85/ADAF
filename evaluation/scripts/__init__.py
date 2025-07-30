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
    consolidate_parser.add_argument("--dataset", required=True, choices=["tatqa", "finqa", "fetaqa", "fetaqa_perturbed", "mmqa"])
    consolidate_parser.add_argument("--run_dir", required=True, help="Path to run directory")
    consolidate_parser.add_argument("--quiet", action="store_true", help="Suppress warnings")
    
    # Postprocess command
    postprocess_parser = subparsers.add_parser("postprocess", help="Post-process consolidated predictions")
    postprocess_parser.add_argument("--dataset", required=True, choices=["tatqa", "finqa", "fetaqa", "fetaqa_perturbed", "mmqa"])
    postprocess_parser.add_argument("--consolidated_file", required=True, help="Path to consolidated JSON")
    
    # Score command
    score_parser = subparsers.add_parser("score", help="Score post-processed predictions")
    score_parser.add_argument("--dataset", required=True, choices=["tatqa", "finqa", "fetaqa", "fetaqa_perturbed", "mmqa"])
    score_parser.add_argument("--post_file", required=True, help="Path to post-processed JSON")
    score_parser.add_argument("--gemini_key", help="Optional Gemini API key")
    score_parser.add_argument("--use_nlg_metrics", action="store_true", help="Calculate NLG metrics (ROUGE-1, BLEU, ROUGE-2, ROUGE-L) for long-form text")
    score_parser.add_argument("--min_nlg_length", type=int, default=45, help="Minimum text length for NLG metrics (default: 45)")
    score_parser.add_argument("--enhanced_nlg_normalization", action="store_true", help="Use enhanced normalization for NLG metrics (lowercase, remove punctuation, etc.)")
    score_parser.add_argument("--optimized_rouge_normalization", action="store_true", help="Use optimized ROUGE normalization (entity standardization, explanatory phrase removal, etc.)")
    score_parser.add_argument("--fetaqa_specific_normalization", action="store_true", default=True, help="Use FETAQA-specific normalization (default: True)")
    
    # Perturbation-specific options
    score_parser.add_argument("--perturbation_class", choices=["invariant", "counterfactual", "unanswerable"], 
                            help="Filter by perturbation class")
    score_parser.add_argument("--perturbation_type", 
                            help="Filter by perturbation type")
    score_parser.add_argument("--detailed_breakdown", action="store_true", 
                            help="Generate detailed class×type breakdown")
    score_parser.add_argument("--robustness_analysis", action="store_true", 
                            help="Include robustness analysis")
    score_parser.add_argument("--output_report", 
                            help="Output file for comprehensive analysis")
    
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
            # Add NLG metrics flag if provided
            if hasattr(args, 'use_nlg_metrics') and args.use_nlg_metrics:
                score_args.append("--use_nlg_metrics")
            if hasattr(args, 'min_nlg_length') and args.min_nlg_length:
                score_args.extend(["--min_nlg_length", str(args.min_nlg_length)])
            if hasattr(args, 'enhanced_nlg_normalization') and args.enhanced_nlg_normalization:
                score_args.append("--enhanced_nlg_normalization")
            if hasattr(args, 'optimized_rouge_normalization') and args.optimized_rouge_normalization:
                score_args.append("--optimized_rouge_normalization")
            if hasattr(args, 'fetaqa_specific_normalization') and args.fetaqa_specific_normalization:
                score_args.append("--fetaqa_specific_normalization")
            score_main(score_args)
        elif args.dataset == "fetaqa_perturbed":
            from .score_run_perturbed_enhanced import main as score_main
            score_args = ["--post_file", args.post_file]
            if args.gemini_key:
                score_args.extend(["--gemini_key", args.gemini_key])
            if hasattr(args, 'use_nlg_metrics') and args.use_nlg_metrics:
                score_args.append("--use_nlg_metrics")
            if hasattr(args, 'min_nlg_length') and args.min_nlg_length:
                score_args.extend(["--min_nlg_length", str(args.min_nlg_length)])
            if hasattr(args, 'enhanced_nlg_normalization') and args.enhanced_nlg_normalization:
                score_args.append("--enhanced_nlg_normalization")
            if hasattr(args, 'optimized_rouge_normalization') and args.optimized_rouge_normalization:
                score_args.append("--optimized_rouge_normalization")
            if hasattr(args, 'fetaqa_specific_normalization') and args.fetaqa_specific_normalization:
                score_args.append("--fetaqa_specific_normalization")
            if hasattr(args, 'perturbation_class') and args.perturbation_class:
                score_args.extend(["--perturbation_class", args.perturbation_class])
            if hasattr(args, 'perturbation_type') and args.perturbation_type:
                score_args.extend(["--perturbation_type", args.perturbation_type])
            if hasattr(args, 'detailed_breakdown') and args.detailed_breakdown:
                score_args.append("--detailed_breakdown")
            if hasattr(args, 'robustness_analysis') and args.robustness_analysis:
                score_args.append("--robustness_analysis")
            if hasattr(args, 'output_report') and args.output_report:
                score_args.extend(["--output_report", args.output_report])
            score_main(score_args)
        else:
            from .score_run import main as score_main
            score_args = ["--post_file", args.post_file]
            if args.gemini_key:
                score_args.extend(["--gemini_key", args.gemini_key])
            score_main(score_args)

if __name__ == "__main__":
    main() 