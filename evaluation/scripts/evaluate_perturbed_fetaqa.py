#!/usr/bin/env python3
"""Evaluate perturbed FETA-QA dataset by perturbation type."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import statistics

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

def evaluate_perturbation_type(run_dir: Path, perturbation_class: str, perturbation_type: str) -> Dict[str, Any]:
    """Evaluate a specific perturbation type."""
    print(f"\n=== Evaluating {perturbation_class}_{perturbation_type} ===")
    
    # Filter files for this perturbation type
    pattern = f"{perturbation_class}_{perturbation_type}_*_out.txt"
    files = list(run_dir.glob(pattern))
    print(f"Found {len(files)} files for {perturbation_class}_{perturbation_type}")
    
    if not files:
        print(f"No files found for {perturbation_class}_{perturbation_type}")
        return {}
    
    # Create a temporary directory with only this perturbation type
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy only the relevant files
        for file in files:
            shutil.copy2(file, temp_path / file.name)
        
        # Run consolidation
        from consolidate_run import main as consolidate_main
        consolidate_args = [
            "--dataset", "fetaqa_perturbed",
            "--run_dir", str(temp_path)
        ]
        
        try:
            consolidate_main(consolidate_args)
            
            # Find the consolidated file
            consolidated_files = list(Path("evaluation/fetaqa_perturbed").rglob("consolidated_*.json"))
            if not consolidated_files:
                print("No consolidated file found")
                return {}
            
            latest_consolidated = max(consolidated_files, key=lambda x: x.stat().st_mtime)
            
            # Run postprocessing
            from postprocess_run import main as postprocess_main
            postprocess_args = [
                "--dataset", "fetaqa_perturbed",
                "--consolidated_file", str(latest_consolidated)
            ]
            postprocess_main(postprocess_args)
            
            # Find the postprocessed file
            postprocessed_files = list(Path("evaluation/fetaqa_perturbed").rglob("postprocessed_*.json"))
            if not postprocessed_files:
                print("No postprocessed file found")
                return {}
            
            latest_postprocessed = max(postprocessed_files, key=lambda x: x.stat().st_mtime)
            
            # Run scoring
            import subprocess
            import sys
            
            score_cmd = [
                sys.executable, "-m", "evaluation.scripts", "score",
                "--dataset", "fetaqa",
                "--post_file", str(latest_postprocessed),
                "--use_nlg_metrics"
            ]
            
            result = subprocess.run(score_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Scoring failed: {result.stderr}")
                return {}
            
            # Find the scored file
            scored_files = list(Path("evaluation/fetaqa_perturbed").rglob("avg_scores_*.json"))
            if not scored_files:
                print("No scored file found")
                return {}
            
            latest_scored = max(scored_files, key=lambda x: x.stat().st_mtime)
            
            # Load results
            with open(latest_scored, 'r') as f:
                results = json.load(f)
            
            # Add metadata
            results["perturbation_class"] = perturbation_class
            results["perturbation_type"] = perturbation_type
            results["total_files"] = len(files)
            
            return results
            
        except Exception as e:
            print(f"Error evaluating {perturbation_class}_{perturbation_type}: {e}")
            return {}

def main():
    parser = argparse.ArgumentParser(description="Evaluate perturbed FETA-QA by perturbation type")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}")
        return
    
    # Define perturbation types
    perturbation_types = [
        ("counterfactual", "change_numeric_cell"),
        ("counterfactual", "replace_entity_context"),
        ("counterfactual", "replace_entity_question"),
        ("invariant", "paraphrase_context"),
        ("invariant", "paraphrase_question"),
        ("invariant", "shuffle_rows"),
        ("unanswerable", "ask_missing_entity"),
        ("unanswerable", "remove_key_context"),
        ("unanswerable", "remove_relevant_row")
    ]
    
    results = {}
    all_scores = []
    
    for perturbation_class, perturbation_type in perturbation_types:
        result = evaluate_perturbation_type(run_dir, perturbation_class, perturbation_type)
        if result:
            key = f"{perturbation_class}_{perturbation_type}"
            results[key] = result
            all_scores.append(result)
    
    # Calculate averages
    if all_scores:
        overall_avg = {
            "exact_match": statistics.mean([r["overall"]["exact"] for r in all_scores]),
            "f1": statistics.mean([r["overall"]["f1"] for r in all_scores]),
            "rouge_1": statistics.mean([r["overall"]["rouge_1"] for r in all_scores]),
            "rouge_2": statistics.mean([r["overall"]["rouge_2"] for r in all_scores]),
            "rouge_l": statistics.mean([r["overall"]["rouge_l"] for r in all_scores]),
            "bleu": statistics.mean([r["overall"]["bleu"] for r in all_scores]),
            "total_examples": sum([r["overall"]["count"] for r in all_scores])
        }
        
        results["overall_average"] = overall_avg
        
        print(f"\n=== OVERALL AVERAGE ACROSS ALL PERTURBATION TYPES ===")
        print(f"Total Examples: {overall_avg['total_examples']}")
        print(f"Exact Match: {overall_avg['exact_match']:.4f}")
        print(f"F1 Score: {overall_avg['f1']:.4f}")
        print(f"ROUGE-1: {overall_avg['rouge_1']:.4f}")
        print(f"ROUGE-2: {overall_avg['rouge_2']:.4f}")
        print(f"ROUGE-L: {overall_avg['rouge_l']:.4f}")
        print(f"BLEU: {overall_avg['bleu']:.4f}")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"perturbed_fetaqa_evaluation_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main() 