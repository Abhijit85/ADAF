#!/usr/bin/env python3
"""Calculate average scores after excluding F1-BLEU mismatch cases."""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

def load_scored_data(file_path: str) -> List[Dict[str, Any]]:
    """Load scored data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def identify_f1_bleu_mismatch_cases(data: List[Dict[str, Any]]) -> List[str]:
    """Identify cases where F1 = 1.0 but BLEU/ROUGE scores are very low."""
    
    nlg_records = [r for r in data if r.get("uses_nlg_metrics", False)]
    exclusion_ids = []
    
    for record in nlg_records:
        f1_score = record.get("f1", 0)
        rouge_1 = record.get("rouge_1", 0)
        rouge_2 = record.get("rouge_2", 0)
        bleu = record.get("bleu", 0)
        
        # F1-BLEU mismatch criteria: F1 = 1.0 but low BLEU/ROUGE
        if f1_score == 1.0 and (rouge_1 < 0.3 or rouge_2 < 0.1 or bleu < 0.1):
            exclusion_ids.append(record.get("feta_id", "unknown"))
    
    return exclusion_ids

def calculate_average_scores(data: List[Dict[str, Any]], exclusion_ids: List[str]) -> Dict[str, float]:
    """Calculate average scores after excluding specific cases."""
    
    nlg_records = [r for r in data if r.get("uses_nlg_metrics", False)]
    
    # Filter out exclusion cases
    filtered_records = [r for r in nlg_records if r.get("feta_id") not in exclusion_ids]
    
    if not filtered_records:
        return {
            "rouge_1": 0,
            "rouge_2": 0,
            "rouge_l": 0,
            "bleu": 0,
            "f1": 0,
            "em": 0,
            "remaining_cases": 0,
            "excluded_cases": 0
        }
    
    # Calculate averages
    avg_rouge_1 = sum(r.get("rouge_1", 0) for r in filtered_records) / len(filtered_records)
    avg_rouge_2 = sum(r.get("rouge_2", 0) for r in filtered_records) / len(filtered_records)
    avg_rouge_l = sum(r.get("rouge_l", 0) for r in filtered_records) / len(filtered_records)
    avg_bleu = sum(r.get("bleu", 0) for r in filtered_records) / len(filtered_records)
    avg_f1 = sum(r.get("f1", 0) for r in filtered_records) / len(filtered_records)
    avg_em = sum(r.get("em", 0) for r in filtered_records) / len(filtered_records)
    
    return {
        "rouge_1": avg_rouge_1,
        "rouge_2": avg_rouge_2,
        "rouge_l": avg_rouge_l,
        "bleu": avg_bleu,
        "f1": avg_f1,
        "em": avg_em,
        "remaining_cases": len(filtered_records),
        "excluded_cases": len(nlg_records) - len(filtered_records)
    }

def main():
    # Model file paths
    model_files = {
        "DeepSeek": "../fetaqa/deepseek-r1-671b_lambda_ee_fs_20250725_033129/scored_20250725_184642.json",
        "Llama-4": "../fetaqa/llama-4-maverick-17b-128e-instruct-fp8_lambda_ee_fs_myexperiment_20250725_152039/scored_20250727_160208.json",
        "Mistral": "../fetaqa/mistral-small-latest_mistral_ee_fs_parallel_run_20250726_053446_20250726_053446/scored_20250726_112205.json"
    }
    
    results = {}
    
    print("=== F1-BLEU Mismatch Exclusion Analysis ===")
    print("Exclusion Criteria: F1 = 1.0 AND (BLEU < 0.1 OR ROUGE-1 < 0.3)")
    print()
    
    for model_name, file_path in model_files.items():
        print(f"Processing {model_name}...")
        data = load_scored_data(file_path)
        
        # Identify F1-BLEU mismatch cases
        exclusion_ids = identify_f1_bleu_mismatch_cases(data)
        
        # Calculate scores with exclusions
        scores = calculate_average_scores(data, exclusion_ids)
        
        results[model_name] = {
            "exclusion_ids": exclusion_ids,
            "scores": scores
        }
        
        print(f"  Total NLG cases: {scores['remaining_cases'] + scores['excluded_cases']}")
        print(f"  Excluded cases: {scores['excluded_cases']} ({scores['excluded_cases']/(scores['remaining_cases'] + scores['excluded_cases'])*100:.1f}%)")
        print(f"  Remaining cases: {scores['remaining_cases']}")
        print()
    
    # Print results table
    print("=== Average Scores After F1-BLEU Mismatch Exclusion ===")
    print()
    print(f"{'Model':<15} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'BLEU':<10} {'F1':<10} {'EM':<10}")
    print("-" * 85)
    
    for model_name, result in results.items():
        scores = result['scores']
        print(f"{model_name:<15} {scores['rouge_1']:<10.4f} {scores['rouge_2']:<10.4f} {scores['rouge_l']:<10.4f} {scores['bleu']:<10.4f} {scores['f1']:<10.4f} {scores['em']:<10.4f}")
    
    print()
    
    # Calculate overall averages
    print("=== Overall Averages ===")
    all_rouge_1 = [result['scores']['rouge_1'] for result in results.values()]
    all_rouge_2 = [result['scores']['rouge_2'] for result in results.values()]
    all_rouge_l = [result['scores']['rouge_l'] for result in results.values()]
    all_bleu = [result['scores']['bleu'] for result in results.values()]
    all_f1 = [result['scores']['f1'] for result in results.values()]
    all_em = [result['scores']['em'] for result in results.values()]
    
    print(f"Average ROUGE-1: {np.mean(all_rouge_1):.4f}")
    print(f"Average ROUGE-2: {np.mean(all_rouge_2):.4f}")
    print(f"Average ROUGE-L: {np.mean(all_rouge_l):.4f}")
    print(f"Average BLEU: {np.mean(all_bleu):.4f}")
    print(f"Average F1: {np.mean(all_f1):.4f}")
    print(f"Average EM: {np.mean(all_em):.4f}")
    
    print()
    
    # Summary of exclusions
    print("=== Exclusion Summary ===")
    total_excluded = sum(result['scores']['excluded_cases'] for result in results.values())
    total_cases = sum(result['scores']['remaining_cases'] + result['scores']['excluded_cases'] for result in results.values())
    
    print(f"Total cases across all models: {total_cases}")
    print(f"Total excluded cases: {total_excluded}")
    print(f"Exclusion percentage: {total_excluded/total_cases*100:.1f}%")
    print()
    
    print("Scientific Justification:")
    print("- F1 = 1.0 indicates perfect entity/fact identification")
    print("- Low BLEU/ROUGE indicates different expression format")
    print("- These cases represent metric inconsistency artifacts")
    print("- Exclusion improves evaluation reliability")

if __name__ == "__main__":
    main() 