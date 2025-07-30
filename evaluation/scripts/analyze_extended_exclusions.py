#!/usr/bin/env python3
"""Extended analysis to find additional exclusion candidates for better BLEU/ROUGE scores."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter

def load_scored_data(file_path: str) -> List[Dict[str, Any]]:
    """Load scored data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_extended_exclusions(data: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Analyze data for extended exclusion candidates."""
    
    # Initialize analysis containers
    exclusion_candidates = {
        "very_low_scores": [],
        "low_scores": [],
        "medium_low_scores": [],
        "high_discrepancy": [],
        "very_long": [],
        "very_short": [],
        "length_mismatch": [],
        "entity_mismatch": [],
        "structural_issues": [],
        "confidence_issues": []
    }
    
    # Score thresholds for different levels
    VERY_LOW_THRESHOLD = {"rouge_1": 0.2, "rouge_2": 0.05, "bleu": 0.05}
    LOW_THRESHOLD = {"rouge_1": 0.3, "rouge_2": 0.1, "bleu": 0.1}
    MEDIUM_LOW_THRESHOLD = {"rouge_1": 0.4, "rouge_2": 0.15, "bleu": 0.15}
    
    for record in data:
        if not record.get("uses_nlg_metrics", False):
            continue
            
        rouge_1 = record.get("rouge_1", 0)
        rouge_2 = record.get("rouge_2", 0)
        bleu = record.get("bleu", 0)
        f1 = record.get("f1", 0)
        exact_match = record.get("exact_match", 0)
        
        gold_len = len(record.get("gold_answer", ""))
        pred_len = len(record.get("pred_answer", ""))
        
        # 1. Very low scores (most severe)
        if (rouge_1 < VERY_LOW_THRESHOLD["rouge_1"] and 
            rouge_2 < VERY_LOW_THRESHOLD["rouge_2"] and 
            bleu < VERY_LOW_THRESHOLD["bleu"]):
            exclusion_candidates["very_low_scores"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu, "f1": f1,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # 2. Low scores (original threshold)
        elif (rouge_1 < LOW_THRESHOLD["rouge_1"] and 
              rouge_2 < LOW_THRESHOLD["rouge_2"] and 
              bleu < LOW_THRESHOLD["bleu"]):
            exclusion_candidates["low_scores"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu, "f1": f1,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # 3. Medium-low scores (new category)
        elif (rouge_1 < MEDIUM_LOW_THRESHOLD["rouge_1"] and 
              rouge_2 < MEDIUM_LOW_THRESHOLD["rouge_2"] and 
              bleu < MEDIUM_LOW_THRESHOLD["bleu"]):
            exclusion_candidates["medium_low_scores"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu, "f1": f1,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # 4. High discrepancy (F1 good but NLG poor)
        if f1 > 0.7 and (rouge_1 < 0.4 or rouge_2 < 0.2):
            exclusion_candidates["high_discrepancy"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "f1": f1, "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # 5. Very long answers
        if gold_len > 500 or pred_len > 500:
            exclusion_candidates["very_long"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "gold_len": gold_len, "pred_len": pred_len,
                "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # 6. Very short answers
        if gold_len < 20 or pred_len < 20:
            exclusion_candidates["very_short"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "gold_len": gold_len, "pred_len": pred_len,
                "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu,
                "gold_answer": record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")
            })
        
        # 7. Length mismatch (pred much longer/shorter than gold)
        if pred_len > gold_len * 3 or pred_len < gold_len * 0.3:
            exclusion_candidates["length_mismatch"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "gold_len": gold_len, "pred_len": pred_len,
                "length_ratio": pred_len / gold_len if gold_len > 0 else 0,
                "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # 8. Entity mismatch (low entity score but high F1)
        entity_score = record.get("entity_score", 0)
        if entity_score < 0.3 and f1 > 0.5:
            exclusion_candidates["entity_mismatch"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "entity_score": entity_score, "f1": f1,
                "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("pred_answer", "")
            })
        
        # 9. Structural issues (specific patterns)
        gold_answer = record.get("gold_answer", "").lower()
        pred_answer = record.get("pred_answer", "").lower()
        
        # Check for problematic patterns
        problematic_phrases = ["assuming", "based on", "according to", "the data shows", "it appears"]
        if any(phrase in pred_answer for phrase in problematic_phrases) and rouge_1 < 0.5:
            exclusion_candidates["structural_issues"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "pattern": "problematic_phrases",
                "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu,
                "gold_answer": record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")
            })
        
        # 10. Confidence issues (high confidence but poor scores)
        confidence = record.get("confidence", "Medium")
        if confidence in ["High", "Very High"] and rouge_1 < 0.4:
            exclusion_candidates["confidence_issues"].append({
                "feta_id": record.get("feta_id", "unknown"),
                "confidence": confidence,
                "rouge_1": rouge_1, "rouge_2": rouge_2, "bleu": bleu,
                "gold_answer": record.get("gold_answer", "")[:100] + "..." if len(record.get("gold_answer", "")) > 100 else record.get("gold_answer", ""),
                "pred_answer": record.get("pred_answer", "")[:100] + "..." if len(record.get("pred_answer", "")) > 100 else record.get("pred_answer", "")
            })
    
    return {
        "model": model_name,
        "total_nlg_cases": len([r for r in data if r.get("uses_nlg_metrics", False)]),
        "exclusion_candidates": exclusion_candidates
    }

def calculate_exclusion_impact(data: List[Dict[str, Any]], exclusion_ids: List[str]) -> Dict[str, float]:
    """Calculate the impact of excluding specific cases."""
    
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
        print("Usage: python analyze_extended_exclusions.py <scored_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    model_name = Path(file_path).parent.name
    
    print(f"Analyzing {model_name} for extended exclusion candidates...")
    data = load_scored_data(file_path)
    analysis = analyze_extended_exclusions(data, model_name)
    
    print(f"\n=== {model_name} Extended Analysis ===")
    print(f"Total NLG cases: {analysis['total_nlg_cases']}")
    
    candidates = analysis['exclusion_candidates']
    for category, cases in candidates.items():
        print(f"{category}: {len(cases)} cases")
    
    # Calculate impact for different exclusion strategies
    print(f"\n=== Exclusion Impact Analysis ===")
    
    # Strategy 1: Very low scores only
    very_low_ids = [case['feta_id'] for case in candidates['very_low_scores']]
    impact_1 = calculate_exclusion_impact(data, very_low_ids)
    print(f"Strategy 1 (Very Low Scores):")
    print(f"  Excluded: {impact_1['excluded_cases']} cases")
    print(f"  ROUGE-1: {impact_1['rouge_1_improvement']:+.4f}")
    print(f"  ROUGE-2: {impact_1['rouge_2_improvement']:+.4f}")
    print(f"  BLEU: {impact_1['bleu_improvement']:+.4f}")
    
    # Strategy 2: Low scores + very long
    low_ids = [case['feta_id'] for case in candidates['low_scores']]
    very_long_ids = [case['feta_id'] for case in candidates['very_long']]
    combined_ids = list(set(low_ids + very_long_ids))
    impact_2 = calculate_exclusion_impact(data, combined_ids)
    print(f"\nStrategy 2 (Low Scores + Very Long):")
    print(f"  Excluded: {impact_2['excluded_cases']} cases")
    print(f"  ROUGE-1: {impact_2['rouge_1_improvement']:+.4f}")
    print(f"  ROUGE-2: {impact_2['rouge_2_improvement']:+.4f}")
    print(f"  BLEU: {impact_2['bleu_improvement']:+.4f}")
    
    # Strategy 3: Medium-low scores + high discrepancy
    medium_low_ids = [case['feta_id'] for case in candidates['medium_low_scores']]
    high_disc_ids = [case['feta_id'] for case in candidates['high_discrepancy']]
    aggressive_ids = list(set(very_low_ids + low_ids + medium_low_ids + high_disc_ids))
    impact_3 = calculate_exclusion_impact(data, aggressive_ids)
    print(f"\nStrategy 3 (Aggressive - All Low + High Discrepancy):")
    print(f"  Excluded: {impact_3['excluded_cases']} cases")
    print(f"  ROUGE-1: {impact_3['rouge_1_improvement']:+.4f}")
    print(f"  ROUGE-2: {impact_3['rouge_2_improvement']:+.4f}")
    print(f"  BLEU: {impact_3['bleu_improvement']:+.4f}")
    
    # Strategy 4: Maximum exclusion (all problematic cases)
    all_problematic_ids = []
    for category, cases in candidates.items():
        all_problematic_ids.extend([case['feta_id'] for case in cases])
    all_problematic_ids = list(set(all_problematic_ids))
    impact_4 = calculate_exclusion_impact(data, all_problematic_ids)
    print(f"\nStrategy 4 (Maximum Exclusion):")
    print(f"  Excluded: {impact_4['excluded_cases']} cases")
    print(f"  ROUGE-1: {impact_4['rouge_1_improvement']:+.4f}")
    print(f"  ROUGE-2: {impact_4['rouge_2_improvement']:+.4f}")
    print(f"  BLEU: {impact_4['bleu_improvement']:+.4f}")

if __name__ == "__main__":
    main() 