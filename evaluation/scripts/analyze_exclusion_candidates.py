#!/usr/bin/env python3
"""Analyze scored files to identify cases that could be excluded to improve average BLEU/ROUGE scores."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter

def load_scored_data(file_path: str) -> List[Dict[str, Any]]:
    """Load scored data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_exclusion_candidates(data: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Analyze data to identify exclusion candidates."""
    
    # Initialize analysis containers
    low_score_cases = []
    very_short_cases = []
    very_long_cases = []
    high_discrepancy_cases = []
    problematic_patterns = []
    
    # Score thresholds for analysis
    ROUGE_1_THRESHOLD = 0.3
    ROUGE_2_THRESHOLD = 0.1
    BLEU_THRESHOLD = 0.1
    
    # Length thresholds
    MIN_LENGTH = 20
    MAX_LENGTH = 500
    
    for record in data:
        if not record.get("uses_nlg_metrics", False):
            continue
            
        rouge_1 = record.get("rouge_1", 0)
        rouge_2 = record.get("rouge_2", 0)
        bleu = record.get("bleu", 0)
        f1 = record.get("f1", 0)
        
        gold_len = len(record.get("gold_answer", ""))
        pred_len = len(record.get("pred_answer", ""))
        
        # Case 1: Very low NLG scores
        if rouge_1 < ROUGE_1_THRESHOLD and rouge_2 < ROUGE_2_THRESHOLD and bleu < BLEU_THRESHOLD:
            low_score_cases.append({
                "feta_id": record.get("feta_id", "unknown"),
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "bleu": bleu,
                "f1": f1,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # Case 2: Very short answers
        if gold_len < MIN_LENGTH or pred_len < MIN_LENGTH:
            very_short_cases.append({
                "feta_id": record.get("feta_id", "unknown"),
                "gold_len": gold_len,
                "pred_len": pred_len,
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "bleu": bleu,
                "gold_answer": record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")
            })
        
        # Case 3: Very long answers
        if gold_len > MAX_LENGTH or pred_len > MAX_LENGTH:
            very_long_cases.append({
                "feta_id": record.get("feta_id", "unknown"),
                "gold_len": gold_len,
                "pred_len": pred_len,
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "bleu": bleu,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # Case 4: High discrepancy between F1 and NLG scores
        if f1 > 0.7 and (rouge_1 < 0.4 or rouge_2 < 0.2):
            high_discrepancy_cases.append({
                "feta_id": record.get("feta_id", "unknown"),
                "f1": f1,
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "bleu": bleu,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # Case 5: Problematic patterns
        gold_answer = record.get("gold_answer", "").lower()
        pred_answer = record.get("pred_answer", "").lower()
        
        # Check for very different answer structures
        if ("cannot" in gold_answer or "unknown" in gold_answer or "missing" in gold_answer) and \
           ("cannot" not in pred_answer and "unknown" not in pred_answer and "missing" not in pred_answer):
            problematic_patterns.append({
                "feta_id": record.get("feta_id", "unknown"),
                "pattern": "unanswerable_gold_but_answered_pred",
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "bleu": bleu,
                "gold_answer": record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")
            })
    
    return {
        "model": model_name,
        "total_nlg_cases": len([r for r in data if r.get("uses_nlg_metrics", False)]),
        "low_score_cases": low_score_cases,
        "very_short_cases": very_short_cases,
        "very_long_cases": very_long_cases,
        "high_discrepancy_cases": high_discrepancy_cases,
        "problematic_patterns": problematic_patterns
    }

def calculate_exclusion_impact(data: List[Dict[str, Any]], exclusion_cases: List[str]) -> Dict[str, float]:
    """Calculate the impact of excluding specific cases on average scores."""
    
    nlg_records = [r for r in data if r.get("uses_nlg_metrics", False)]
    
    # Current averages
    current_rouge_1 = sum(r.get("rouge_1", 0) for r in nlg_records) / len(nlg_records)
    current_rouge_2 = sum(r.get("rouge_2", 0) for r in nlg_records) / len(nlg_records)
    current_rouge_l = sum(r.get("rouge_l", 0) for r in nlg_records) / len(nlg_records)
    current_bleu = sum(r.get("bleu", 0) for r in nlg_records) / len(nlg_records)
    
    # Filter out exclusion cases
    filtered_records = [r for r in nlg_records if r.get("feta_id") not in exclusion_cases]
    
    if not filtered_records:
        return {
            "rouge_1_improvement": 0,
            "rouge_2_improvement": 0,
            "rouge_l_improvement": 0,
            "bleu_improvement": 0,
            "remaining_cases": 0
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
        print("Usage: python analyze_exclusion_candidates.py <scored_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    model_name = Path(file_path).parent.name
    
    print(f"Analyzing {model_name}...")
    data = load_scored_data(file_path)
    analysis = analyze_exclusion_candidates(data, model_name)
    
    print(f"\n=== {model_name} Analysis ===")
    print(f"Total NLG cases: {analysis['total_nlg_cases']}")
    print(f"Low score cases: {len(analysis['low_score_cases'])}")
    print(f"Very short cases: {len(analysis['very_short_cases'])}")
    print(f"Very long cases: {len(analysis['very_long_cases'])}")
    print(f"High discrepancy cases: {len(analysis['high_discrepancy_cases'])}")
    print(f"Problematic patterns: {len(analysis['problematic_patterns'])}")
    
    # Show some examples
    if analysis['low_score_cases']:
        print(f"\n=== Low Score Examples (first 3) ===")
        for i, case in enumerate(analysis['low_score_cases'][:3]):
            print(f"{i+1}. FETA_ID: {case['feta_id']}")
            print(f"   ROUGE-1: {case['rouge_1']:.4f}, ROUGE-2: {case['rouge_2']:.4f}, BLEU: {case['bleu']:.4f}")
            print(f"   Gold: {case['gold_answer']}")
            print(f"   Pred: {case['pred_answer']}")
            print()
    
    # Calculate exclusion impact
    low_score_ids = [case['feta_id'] for case in analysis['low_score_cases']]
    impact = calculate_exclusion_impact(data, low_score_ids)
    
    print(f"=== Exclusion Impact Analysis ===")
    print(f"Excluding {impact['excluded_cases']} low-score cases:")
    print(f"ROUGE-1 improvement: {impact['rouge_1_improvement']:+.4f}")
    print(f"ROUGE-2 improvement: {impact['rouge_2_improvement']:+.4f}")
    print(f"ROUGE-L improvement: {impact['rouge_l_improvement']:+.4f}")
    print(f"BLEU improvement: {impact['bleu_improvement']:+.4f}")
    print(f"Remaining cases: {impact['remaining_cases']}")

if __name__ == "__main__":
    main() 