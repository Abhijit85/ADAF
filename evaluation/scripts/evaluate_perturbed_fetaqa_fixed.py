#!/usr/bin/env python3
"""Evaluate perturbed FETA-QA dataset by perturbation type with proper organization."""

import argparse
import json
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import statistics
import tempfile

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

def get_perturbation_mapping() -> List[Tuple[str, str, str]]:
    """Get the mapping of perturbation classes and types based on filename patterns."""
    return [
        ("counterfactual", "change_numeric_cell", "counterfactual_change_numeric"),
        ("counterfactual", "replace_entity_context", "counterfactual_replace_entity_context"),
        ("counterfactual", "replace_entity_question", "counterfactual_replace_entity_question"),
        ("invariant", "paraphrase_context", "invariant_paraphrase_context"),
        ("invariant", "paraphrase_question", "invariant_paraphrase_question"),
        ("invariant", "shuffle_rows", "invariant_shuffle_rows"),
        ("unanswerable", "ask_missing_entity", "unanswerable_ask_missing"),
        ("unanswerable", "remove_key_context", "unanswerable_remove_key"),
        ("unanswerable", "remove_relevant_row", "unanswerable_remove_relevant")
    ]

def evaluate_perturbation_type(run_dir: Path, perturbation_class: str, perturbation_type: str, 
                             filename_pattern: str, output_base_dir: Path) -> Dict[str, Any]:
    """Evaluate a specific perturbation type."""
    print(f"\n=== Evaluating {perturbation_class}_{perturbation_type} ===")
    
    # Filter files for this perturbation type using the filename pattern
    files = list(run_dir.glob(f"{filename_pattern}_*_out.txt"))
    print(f"Found {len(files)} files for {perturbation_class}_{perturbation_type}")
    
    if not files:
        print(f"No files found for {perturbation_class}_{perturbation_type}")
        return {}
    
    # Create output directory for this perturbation type
    output_dir = output_base_dir / f"{perturbation_class}_{perturbation_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory with only this perturbation type
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy only the relevant files
        for file in files:
            shutil.copy2(file, temp_path / file.name)
        
        # Run consolidation
        from consolidate_run import main as consolidate_main
        consolidate_args = [
            "--dataset", "fetaqa_perturbed",
            "--run_dir", str(temp_path),
            "--perturbation_class", perturbation_class,
            "--perturbation_type", perturbation_type
        ]
        
        # Set environment variables to pass perturbation info to the dataset class
        import os
        os.environ["PERTURBATION_CLASS"] = perturbation_class
        os.environ["PERTURBATION_TYPE"] = perturbation_type
        
        try:
            consolidate_main(consolidate_args)
            
            # Find the consolidated file
            consolidated_files = list(Path("evaluation/fetaqa_perturbed").rglob("consolidated_*.json"))
            if not consolidated_files:
                print("No consolidated file found")
                return {}
            
            latest_consolidated = max(consolidated_files, key=lambda x: x.stat().st_mtime)
            
            # Copy consolidated file to output directory
            shutil.copy2(latest_consolidated, output_dir / latest_consolidated.name)
            
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
            
            # Copy postprocessed file to output directory
            shutil.copy2(latest_postprocessed, output_dir / latest_postprocessed.name)
            
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
            
            # Find the scored files
            scored_files = list(Path("evaluation/fetaqa_perturbed").rglob("scored_*.json"))
            avg_scores_files = list(Path("evaluation/fetaqa_perturbed").rglob("avg_scores_*.json"))
            
            if scored_files:
                latest_scored = max(scored_files, key=lambda x: x.stat().st_mtime)
                shutil.copy2(latest_scored, output_dir / latest_scored.name)
            
            if avg_scores_files:
                latest_avg_scores = max(avg_scores_files, key=lambda x: x.stat().st_mtime)
                shutil.copy2(latest_avg_scores, output_dir / latest_avg_scores.name)
                
                # Load results
                with open(latest_avg_scores, 'r') as f:
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
    parser.add_argument("--output_dir", type=str, help="Output directory for organized results")
    
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}")
        return
    
    # Set output directory
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = Path(f"ADAF/ADAF/evaluation/fetaqa_perturbed/organized_results_{timestamp}")
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Get perturbation mappings
    perturbation_mappings = get_perturbation_mapping()
    
    results = {}
    all_scores = []
    
    for perturbation_class, perturbation_type, filename_pattern in perturbation_mappings:
        result = evaluate_perturbation_type(run_dir, perturbation_class, perturbation_type, 
                                         filename_pattern, output_base_dir)
        if result:
            key = f"{perturbation_class}_{perturbation_type}"
            results[key] = result
            all_scores.append(result)
    
    # Calculate averages
    if all_scores:
        # Safely calculate averages, handling missing metrics
        def safe_mean(values, key):
            valid_values = [r["overall"].get(key, 0) for r in all_scores if r.get("overall") and r["overall"].get(key) is not None]
            return statistics.mean(valid_values) if valid_values else 0
        
        overall_avg = {
            "exact_match": safe_mean(all_scores, "exact"),
            "f1": safe_mean(all_scores, "f1"),
            "rouge_1": safe_mean(all_scores, "rouge_1"),
            "rouge_2": safe_mean(all_scores, "rouge_2"),
            "rouge_l": safe_mean(all_scores, "rouge_l"),
            "bleu": safe_mean(all_scores, "bleu"),
            "total_examples": sum([r.get("overall", {}).get("count", 0) for r in all_scores])
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
    
    # Save overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_results_file = output_base_dir / f"overall_results_{timestamp}.json"
    
    with open(overall_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults organized in: {output_base_dir}")
    print(f"Overall results saved to: {overall_results_file}")

if __name__ == "__main__":
    main() 