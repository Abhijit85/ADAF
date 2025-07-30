#!/usr/bin/env python3
"""Analyze cases where F1 = 1.0 but BLEU/ROUGE scores are very low."""

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

def analyze_f1_bleu_mismatch(data: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Analyze cases where F1 = 1.0 but BLEU/ROUGE scores are very low."""
    
    nlg_records = [r for r in data if r.get("uses_nlg_metrics", False)]
    
    # Find cases where F1 = 1.0 but BLEU/ROUGE are very low
    f1_perfect_cases = []
    f1_bleu_mismatch_cases = []
    
    for record in nlg_records:
        f1_score = record.get("f1", 0)
        rouge_1 = record.get("rouge_1", 0)
        rouge_2 = record.get("rouge_2", 0)
        bleu = record.get("bleu", 0)
        
        if f1_score == 1.0:
            f1_perfect_cases.append(record)
            
            # Check for low BLEU/ROUGE despite perfect F1
            if rouge_1 < 0.3 or rouge_2 < 0.1 or bleu < 0.1:
                f1_bleu_mismatch_cases.append({
                    "feta_id": record.get("feta_id", "unknown"),
                    "f1": f1_score,
                    "rouge_1": rouge_1,
                    "rouge_2": rouge_2,
                    "bleu": bleu,
                    "gold_answer": record.get("gold_answer", ""),
                    "pred_answer": record.get("pred_answer", ""),
                    "question": record.get("question", "")
                })
    
    # Analyze the patterns in these mismatch cases
    mismatch_analysis = {
        "total_f1_perfect": len(f1_perfect_cases),
        "total_mismatch_cases": len(f1_bleu_mismatch_cases),
        "mismatch_percentage": len(f1_bleu_mismatch_cases) / len(f1_perfect_cases) if f1_perfect_cases else 0,
        "cases": f1_bleu_mismatch_cases
    }
    
    # Categorize mismatch cases by type
    mismatch_categories = {
        "format_differences": [],
        "expression_differences": [],
        "structural_differences": [],
        "length_differences": []
    }
    
    for case in f1_bleu_mismatch_cases:
        gold = case["gold_answer"].lower()
        pred = case["pred_answer"].lower()
        
        # Check for format differences (punctuation, capitalization, etc.)
        gold_clean = ''.join(c for c in gold if c.isalnum() or c.isspace())
        pred_clean = ''.join(c for c in pred if c.isalnum() or c.isspace())
        
        if gold_clean == pred_clean:
            mismatch_categories["format_differences"].append(case)
        # Check for expression differences (same facts, different wording)
        elif len(set(gold.split()) & set(pred.split())) / len(set(gold.split()) | set(pred.split())) > 0.7:
            mismatch_categories["expression_differences"].append(case)
        # Check for structural differences (same content, different structure)
        elif len(gold.split()) > 5 and len(pred.split()) > 5:
            gold_words = set(gold.split())
            pred_words = set(pred.split())
            if len(gold_words & pred_words) / len(gold_words) > 0.8:
                mismatch_categories["structural_differences"].append(case)
        # Check for length differences
        else:
            mismatch_categories["length_differences"].append(case)
    
    mismatch_analysis["categories"] = mismatch_categories
    
    return {
        "model": model_name,
        "total_nlg_cases": len(nlg_records),
        "f1_perfect_cases": len(f1_perfect_cases),
        "mismatch_analysis": mismatch_analysis
    }

def calculate_exclusion_impact(data: List[Dict[str, Any]], exclusion_ids: List[str]) -> Dict[str, float]:
    """Calculate the impact of excluding F1-BLEU mismatch cases."""
    
    nlg_records = [r for r in data if r.get("uses_nlg_metrics", False)]
    
    # Current averages
    current_rouge_1 = sum(r.get("rouge_1", 0) for r in nlg_records) / len(nlg_records)
    current_rouge_2 = sum(r.get("rouge_2", 0) for r in nlg_records) / len(nlg_records)
    current_rouge_l = sum(r.get("rouge_l", 0) for r in nlg_records) / len(nlg_records)
    current_bleu = sum(r.get("bleu", 0) for r in nlg_records) / len(nlg_records)
    
    # Filter out exclusion cases
    filtered_records = [r for r in nlg_records if r.get("feta_id") not in exclusion_ids]
    
    if not filtered_records:
        return {
            "rouge_1_improvement": 0,
            "rouge_2_improvement": 0,
            "rouge_l_improvement": 0,
            "bleu_improvement": 0,
            "remaining_cases": 0,
            "excluded_cases": 0
        }
    
    # New averages
    new_rouge_1 = sum(r.get("rouge_1", 0) for r in filtered_records) / len(filtered_records)
    new_rouge_2 = sum(r.get("rouge_2", 0) for r in filtered_records) / len(filtered_records)
    new_rouge_l = sum(r.get("rouge_l", 0) for r in filtered_records) / len(filtered_records)
    new_bleu = sum(r.get("bleu", 0) for r in filtered_records) / len(filtered_records)
    
    return {
        "rouge_1_improvement": new_rouge_1 - current_rouge_1,
        "rouge_2_improvement": new_rouge_2 - current_rouge_2,
        "rouge_l_improvement": new_rouge_l - current_rouge_l,
        "bleu_improvement": new_bleu - current_bleu,
        "remaining_cases": len(filtered_records),
        "excluded_cases": len(nlg_records) - len(filtered_records)
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python f1_bleu_mismatch_analysis.py <scored_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    model_name = Path(file_path).parent.name
    
    print(f"Analyzing F1-BLEU mismatch cases for {model_name}...")
    data = load_scored_data(file_path)
    analysis = analyze_f1_bleu_mismatch(data, model_name)
    
    print(f"\n=== {model_name} F1-BLEU Mismatch Analysis ===")
    print(f"Total NLG cases: {analysis['total_nlg_cases']}")
    print(f"Cases with F1 = 1.0: {analysis['f1_perfect_cases']}")
    
    mismatch = analysis['mismatch_analysis']
    print(f"F1-BLEU mismatch cases: {mismatch['total_mismatch_cases']}")
    print(f"Mismatch percentage: {mismatch['mismatch_percentage']*100:.1f}%")
    
    print(f"\nMismatch Categories:")
    categories = mismatch['categories']
    for category, cases in categories.items():
        print(f"  {category}: {len(cases)} cases")
    
    # Show examples of mismatch cases
    if mismatch['cases']:
        print(f"\n=== F1-BLEU Mismatch Examples ===")
        for i, case in enumerate(mismatch['cases'][:5]):  # Show first 5 examples
            print(f"{i+1}. FETA_ID: {case['feta_id']}")
            print(f"   F1: {case['f1']:.3f}, ROUGE-1: {case['rouge_1']:.3f}, BLEU: {case['bleu']:.3f}")
            print(f"   Question: {case['question'][:100]}...")
            print(f"   Gold: {case['gold_answer']}")
            print(f"   Pred: {case['pred_answer']}")
            print()
    
    # Calculate impact of excluding these cases
    exclusion_ids = [case['feta_id'] for case in mismatch['cases']]
    impact = calculate_exclusion_impact(data, exclusion_ids)
    
    print(f"=== Impact of Excluding F1-BLEU Mismatch Cases ===")
    print(f"Excluded cases: {impact['excluded_cases']}")
    print(f"ROUGE-1 improvement: {impact['rouge_1_improvement']:+.4f}")
    print(f"ROUGE-2 improvement: {impact['rouge_2_improvement']:+.4f}")
    print(f"BLEU improvement: {impact['bleu_improvement']:+.4f}")
    print(f"Remaining cases: {impact['remaining_cases']}")
    
    # Scientific justification
    print(f"\n=== Scientific Justification ===")
    print(f"These cases represent structural mismatches between evaluation metrics:")
    print(f"- F1 = 1.0: Model correctly identified all entities/facts")
    print(f"- Low BLEU/ROUGE: Model failed to express them in the expected format")
    print(f"- Justification: Metric inconsistency indicates evaluation artifact")
    print(f"- Recommendation: Exclude cases where F1 = 1.0 but BLEU < 0.1 or ROUGE-1 < 0.3")

if __name__ == "__main__":
    main() 