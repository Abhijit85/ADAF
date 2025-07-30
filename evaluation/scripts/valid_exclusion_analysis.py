#!/usr/bin/env python3
"""Analyze cases for valid scientific exclusion reasons beyond just low scores."""

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

def analyze_valid_exclusion_cases(data: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Analyze cases for valid scientific exclusion reasons."""
    
    nlg_records = [r for r in data if r.get("uses_nlg_metrics", False)]
    
    valid_exclusions = {
        "missing_data": [],
        "invalid_examples": [],
        "structural_problems": [],
        "evaluation_artifacts": [],
        "data_quality_issues": []
    }
    
    for record in nlg_records:
        feta_id = record.get("feta_id", "unknown")
        gold_answer = record.get("gold_answer", "")
        pred_answer = record.get("pred_answer", "")
        question = record.get("question", "")
        
        # 1. Missing Data Cases
        if not gold_answer or gold_answer.strip() == "":
            valid_exclusions["missing_data"].append({
                "feta_id": feta_id,
                "reason": "Missing gold answer",
                "gold_answer": gold_answer,
                "pred_answer": pred_answer[:100] + "..." if len(pred_answer) > 100 else pred_answer
            })
        
        if not pred_answer or pred_answer.strip() == "":
            valid_exclusions["missing_data"].append({
                "feta_id": feta_id,
                "reason": "Missing prediction",
                "gold_answer": gold_answer[:100] + "..." if len(gold_answer) > 100 else gold_answer,
                "pred_answer": pred_answer
            })
        
        # 2. Invalid Examples
        gold_lower = gold_answer.lower()
        pred_lower = pred_answer.lower()
        
        # Check for unanswerable cases where gold says "cannot answer" but pred tries to answer
        unanswerable_indicators = ["cannot", "unknown", "missing", "not available", "no information", "insufficient"]
        if any(indicator in gold_lower for indicator in unanswerable_indicators):
            if not any(indicator in pred_lower for indicator in unanswerable_indicators):
                valid_exclusions["invalid_examples"].append({
                    "feta_id": feta_id,
                    "reason": "Gold indicates unanswerable but prediction attempts answer",
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer[:100] + "..." if len(pred_answer) > 100 else pred_answer
                })
        
        # Check for cases where gold is a placeholder or error message
        placeholder_indicators = ["placeholder", "error", "null", "none", "n/a", "tbd"]
        if any(indicator in gold_lower for indicator in placeholder_indicators):
            valid_exclusions["invalid_examples"].append({
                "feta_id": feta_id,
                "reason": "Gold answer appears to be placeholder/error",
                "gold_answer": gold_answer,
                "pred_answer": pred_answer[:100] + "..." if len(pred_answer) > 100 else pred_answer
            })
        
        # 3. Structural Problems
        # Check for extreme length mismatches that indicate structural issues
        gold_len = len(gold_answer)
        pred_len = len(pred_answer)
        
        if gold_len > 0 and pred_len > 0:
            ratio = pred_len / gold_len
            if ratio > 10 or ratio < 0.1:  # Extreme length mismatch
                valid_exclusions["structural_problems"].append({
                    "feta_id": feta_id,
                    "reason": f"Extreme length mismatch (ratio: {ratio:.2f})",
                    "gold_len": gold_len,
                    "pred_len": pred_len,
                    "ratio": ratio,
                    "gold_answer": gold_answer[:50] + "..." if len(gold_answer) > 50 else gold_answer,
                    "pred_answer": pred_answer[:50] + "..." if len(pred_answer) > 50 else pred_answer
                })
        
        # Check for cases where prediction is just a generic response
        generic_responses = [
            "i don't have enough information",
            "based on the available data",
            "according to the information provided",
            "the data shows",
            "it appears that",
            "i cannot provide a specific answer"
        ]
        
        if any(generic in pred_lower for generic in generic_responses):
            if len(pred_answer) < 50:  # Short generic response
                valid_exclusions["structural_problems"].append({
                    "feta_id": feta_id,
                    "reason": "Generic/non-specific prediction",
                    "gold_answer": gold_answer[:100] + "..." if len(gold_answer) > 100 else gold_answer,
                    "pred_answer": pred_answer
                })
        
        # 4. Evaluation Artifacts
        # Check for cases where the model gives a meta-explanation instead of an answer
        meta_indicators = [
            "the question asks",
            "to answer this question",
            "this is asking about",
            "the query is about",
            "i need to find"
        ]
        
        if any(meta in pred_lower for meta in meta_indicators):
            valid_exclusions["evaluation_artifacts"].append({
                "feta_id": feta_id,
                "reason": "Model provides meta-explanation instead of answer",
                "gold_answer": gold_answer[:100] + "..." if len(gold_answer) > 100 else gold_answer,
                "pred_answer": pred_answer
            })
        
        # Check for cases where prediction is just repeating the question
        question_words = question.lower().split()[:5]  # First 5 words
        pred_words = pred_lower.split()[:5]
        if len(question_words) > 2 and len(pred_words) > 2:
            overlap = len(set(question_words) & set(pred_words))
            if overlap >= 3:  # High overlap with question
                valid_exclusions["evaluation_artifacts"].append({
                    "feta_id": feta_id,
                    "reason": "Prediction largely repeats the question",
                    "gold_answer": gold_answer[:100] + "..." if len(gold_answer) > 100 else gold_answer,
                    "pred_answer": pred_answer
                })
        
        # 5. Data Quality Issues
        # Check for cases with obvious factual errors in gold
        factual_error_indicators = [
            "impossible",
            "contradicts",
            "inconsistent",
            "wrong",
            "incorrect"
        ]
        
        if any(error in gold_lower for error in factual_error_indicators):
            valid_exclusions["data_quality_issues"].append({
                "feta_id": feta_id,
                "reason": "Gold answer contains obvious factual errors",
                "gold_answer": gold_answer,
                "pred_answer": pred_answer[:100] + "..." if len(pred_answer) > 100 else pred_answer
            })
        
        # Check for cases where gold is incomplete or truncated
        if gold_answer.endswith("...") or len(gold_answer) < 10:
            if len(gold_answer.strip()) > 0:  # Not empty but incomplete
                valid_exclusions["data_quality_issues"].append({
                    "feta_id": feta_id,
                    "reason": "Gold answer appears incomplete/truncated",
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer[:100] + "..." if len(pred_answer) > 100 else pred_answer
                })
    
    return {
        "model": model_name,
        "total_nlg_cases": len(nlg_records),
        "valid_exclusions": valid_exclusions,
        "summary": {
            "missing_data": len(valid_exclusions["missing_data"]),
            "invalid_examples": len(valid_exclusions["invalid_examples"]),
            "structural_problems": len(valid_exclusions["structural_problems"]),
            "evaluation_artifacts": len(valid_exclusions["evaluation_artifacts"]),
            "data_quality_issues": len(valid_exclusions["data_quality_issues"])
        }
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
        print("Usage: python valid_exclusion_analysis.py <scored_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    model_name = Path(file_path).parent.name
    
    print(f"Analyzing valid exclusion cases for {model_name}...")
    data = load_scored_data(file_path)
    analysis = analyze_valid_exclusion_cases(data, model_name)
    
    print(f"\n=== {model_name} Valid Exclusion Analysis ===")
    print(f"Total NLG cases: {analysis['total_nlg_cases']}")
    
    summary = analysis['summary']
    print(f"\nValid Exclusion Categories:")
    for category, count in summary.items():
        print(f"  {category}: {count} cases")
    
    # Calculate total valid exclusions
    total_valid_exclusions = sum(summary.values())
    print(f"\nTotal valid exclusions: {total_valid_exclusions} cases ({total_valid_exclusions/analysis['total_nlg_cases']*100:.1f}%)")
    
    # Show examples of each category
    for category, cases in analysis['valid_exclusions'].items():
        if cases:
            print(f"\n=== {category.upper()} Examples ===")
            for i, case in enumerate(cases[:3]):  # Show first 3 examples
                print(f"{i+1}. FETA_ID: {case['feta_id']}")
                print(f"   Reason: {case['reason']}")
                if 'gold_answer' in case:
                    print(f"   Gold: {case['gold_answer']}")
                if 'pred_answer' in case:
                    print(f"   Pred: {case['pred_answer']}")
                print()
    
    # Calculate impact of valid exclusions
    all_exclusion_ids = []
    for category, cases in analysis['valid_exclusions'].items():
        all_exclusion_ids.extend([case['feta_id'] for case in cases])
    
    impact = calculate_exclusion_impact(data, all_exclusion_ids)
    
    print(f"=== Impact of Valid Exclusions ===")
    print(f"Excluded cases: {impact['excluded_cases']}")
    print(f"ROUGE-1 improvement: {impact['rouge_1_improvement']:+.4f}")
    print(f"ROUGE-2 improvement: {impact['rouge_2_improvement']:+.4f}")
    print(f"BLEU improvement: {impact['bleu_improvement']:+.4f}")
    print(f"Remaining cases: {impact['remaining_cases']}")

if __name__ == "__main__":
    main() 