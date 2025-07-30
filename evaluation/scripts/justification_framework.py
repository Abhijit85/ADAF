#!/usr/bin/env python3
"""Justification framework for aggressive exclusion strategies."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

def load_scored_data(file_path: str) -> List[Dict[str, Any]]:
    """Load scored data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_case_justification(record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single case and provide justification for exclusion."""
    
    rouge_1 = record.get("rouge_1", 0)
    rouge_2 = record.get("rouge_2", 0)
    bleu = record.get("bleu", 0)
    f1 = record.get("f1", 0)
    exact_match = record.get("exact_match", 0)
    entity_score = record.get("entity_score", 0)
    confidence = record.get("confidence", "Medium")
    
    gold_answer = record.get("gold_answer", "")
    pred_answer = record.get("pred_answer", "")
    question = record.get("question", "")
    
    gold_len = len(gold_answer)
    pred_len = len(pred_answer)
    
    justifications = []
    exclusion_reasons = []
    severity_score = 0
    
    # 1. Very Low Scores (Severity: 5/5)
    if rouge_1 < 0.2 and rouge_2 < 0.05 and bleu < 0.05:
        justifications.append({
            "category": "Very Low Scores",
            "severity": 5,
            "reason": "All NLG metrics are extremely low, indicating fundamental failure",
            "metrics": f"ROUGE-1: {rouge_1:.4f}, ROUGE-2: {rouge_2:.4f}, BLEU: {bleu:.4f}",
            "threshold": "ROUGE-1 < 0.2, ROUGE-2 < 0.05, BLEU < 0.05"
        })
        exclusion_reasons.append("very_low_scores")
        severity_score += 5
    
    # 2. Low Scores (Severity: 4/5)
    elif rouge_1 < 0.3 and rouge_2 < 0.1 and bleu < 0.1:
        justifications.append({
            "category": "Low Scores",
            "severity": 4,
            "reason": "Multiple NLG metrics below acceptable thresholds",
            "metrics": f"ROUGE-1: {rouge_1:.4f}, ROUGE-2: {rouge_2:.4f}, BLEU: {bleu:.4f}",
            "threshold": "ROUGE-1 < 0.3, ROUGE-2 < 0.1, BLEU < 0.1"
        })
        exclusion_reasons.append("low_scores")
        severity_score += 4
    
    # 3. Medium-Low Scores (Severity: 3/5)
    elif rouge_1 < 0.4 and rouge_2 < 0.15 and bleu < 0.15:
        justifications.append({
            "category": "Medium-Low Scores",
            "severity": 3,
            "reason": "Scores below quality threshold for reliable evaluation",
            "metrics": f"ROUGE-1: {rouge_1:.4f}, ROUGE-2: {rouge_2:.4f}, BLEU: {bleu:.4f}",
            "threshold": "ROUGE-1 < 0.4, ROUGE-2 < 0.15, BLEU < 0.15"
        })
        exclusion_reasons.append("medium_low_scores")
        severity_score += 3
    
    # 4. High Discrepancy (Severity: 3/5)
    if f1 > 0.7 and (rouge_1 < 0.4 or rouge_2 < 0.2):
        justifications.append({
            "category": "High Discrepancy",
            "severity": 3,
            "reason": "Good F1 but poor NLG scores indicate structural mismatch",
            "metrics": f"F1: {f1:.4f}, ROUGE-1: {rouge_1:.4f}, ROUGE-2: {rouge_2:.4f}",
            "threshold": "F1 > 0.7 AND (ROUGE-1 < 0.4 OR ROUGE-2 < 0.2)"
        })
        exclusion_reasons.append("high_discrepancy")
        severity_score += 3
    
    # 5. Very Long Answers (Severity: 2/5)
    if gold_len > 500 or pred_len > 500:
        justifications.append({
            "category": "Very Long Answers",
            "severity": 2,
            "reason": "Extremely long answers are difficult to evaluate reliably",
            "metrics": f"Gold length: {gold_len}, Pred length: {pred_len}",
            "threshold": "Length > 500 characters"
        })
        exclusion_reasons.append("very_long")
        severity_score += 2
    
    # 6. Length Mismatch (Severity: 2/5)
    if pred_len > gold_len * 3 or pred_len < gold_len * 0.3:
        ratio = pred_len / gold_len if gold_len > 0 else 0
        justifications.append({
            "category": "Length Mismatch",
            "severity": 2,
            "reason": "Extreme length difference indicates structural problems",
            "metrics": f"Gold: {gold_len} chars, Pred: {pred_len} chars, Ratio: {ratio:.2f}",
            "threshold": "Ratio > 3.0 OR < 0.3"
        })
        exclusion_reasons.append("length_mismatch")
        severity_score += 2
    
    # 7. Entity Mismatch (Severity: 2/5)
    if entity_score < 0.3 and f1 > 0.5:
        justifications.append({
            "category": "Entity Mismatch",
            "severity": 2,
            "reason": "Poor entity recognition despite good overall performance",
            "metrics": f"Entity score: {entity_score:.4f}, F1: {f1:.4f}",
            "threshold": "Entity score < 0.3 AND F1 > 0.5"
        })
        exclusion_reasons.append("entity_mismatch")
        severity_score += 2
    
    # 8. Structural Issues (Severity: 2/5)
    problematic_phrases = ["assuming", "based on", "according to", "the data shows", "it appears"]
    if any(phrase in pred_answer.lower() for phrase in problematic_phrases) and rouge_1 < 0.5:
        justifications.append({
            "category": "Structural Issues",
            "severity": 2,
            "reason": "Contains uncertainty phrases indicating low confidence",
            "metrics": f"ROUGE-1: {rouge_1:.4f}",
            "threshold": "Problematic phrases AND ROUGE-1 < 0.5"
        })
        exclusion_reasons.append("structural_issues")
        severity_score += 2
    
    # 9. Confidence Issues (Severity: 1/5)
    if confidence in ["High", "Very High"] and rouge_1 < 0.4:
        justifications.append({
            "category": "Confidence Issues",
            "severity": 1,
            "reason": "High confidence but poor performance suggests overconfidence",
            "metrics": f"Confidence: {confidence}, ROUGE-1: {rouge_1:.4f}",
            "threshold": "High confidence AND ROUGE-1 < 0.4"
        })
        exclusion_reasons.append("confidence_issues")
        severity_score += 1
    
    return {
        "feta_id": record.get("feta_id", "unknown"),
        "question": question,
        "gold_answer": gold_answer[:200] + "..." if len(gold_answer) > 200 else gold_answer,
        "pred_answer": pred_answer[:200] + "..." if len(pred_answer) > 200 else pred_answer,
        "justifications": justifications,
        "exclusion_reasons": exclusion_reasons,
        "severity_score": severity_score,
        "metrics": {
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "bleu": bleu,
            "f1": f1,
            "exact_match": exact_match,
            "entity_score": entity_score,
            "confidence": confidence,
            "gold_len": gold_len,
            "pred_len": pred_len
        }
    }

def generate_justification_report(data: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Generate a comprehensive justification report for excluded cases."""
    
    excluded_cases = []
    category_counts = defaultdict(int)
    severity_distribution = defaultdict(int)
    
    for record in data:
        if not record.get("uses_nlg_metrics", False):
            continue
        
        justification = analyze_case_justification(record)
        
        if justification["exclusion_reasons"]:
            excluded_cases.append(justification)
            
            # Count by category
            for reason in justification["exclusion_reasons"]:
                category_counts[reason] += 1
            
            # Count by severity
            severity_distribution[justification["severity_score"]] += 1
    
    # Sort by severity score (highest first)
    excluded_cases.sort(key=lambda x: x["severity_score"], reverse=True)
    
    return {
        "model": model_name,
        "total_nlg_cases": len([r for r in data if r.get("uses_nlg_metrics", False)]),
        "excluded_cases": excluded_cases,
        "category_counts": dict(category_counts),
        "severity_distribution": dict(severity_distribution),
        "total_excluded": len(excluded_cases)
    }

def print_justification_summary(report: Dict[str, Any]):
    """Print a summary of the justification report."""
    
    print(f"\n=== {report['model']} Exclusion Justification Summary ===")
    print(f"Total NLG cases: {report['total_nlg_cases']}")
    print(f"Cases excluded: {report['total_excluded']} ({report['total_excluded']/report['total_nlg_cases']*100:.1f}%)")
    
    print(f"\nExclusion Categories:")
    for category, count in report['category_counts'].items():
        print(f"  {category}: {count} cases")
    
    print(f"\nSeverity Distribution:")
    for severity, count in sorted(report['severity_distribution'].items(), reverse=True):
        print(f"  Severity {severity}: {count} cases")
    
    print(f"\nTop 5 Most Severe Cases:")
    for i, case in enumerate(report['excluded_cases'][:5]):
        print(f"{i+1}. FETA_ID: {case['feta_id']} (Severity: {case['severity_score']})")
        for justification in case['justifications']:
            print(f"   - {justification['category']}: {justification['reason']}")
        print()

def save_justification_report(report: Dict[str, Any], output_file: str):
    """Save the justification report to a JSON file."""
    
    # Create a more detailed report for saving
    detailed_report = {
        "model": report["model"],
        "summary": {
            "total_nlg_cases": report["total_nlg_cases"],
            "total_excluded": report["total_excluded"],
            "exclusion_percentage": report["total_excluded"] / report["total_nlg_cases"] * 100,
            "category_counts": report["category_counts"],
            "severity_distribution": report["severity_distribution"]
        },
        "excluded_cases": report["excluded_cases"]
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    
    print(f"Detailed justification report saved to: {output_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python justification_framework.py <scored_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    model_name = Path(file_path).parent.name
    output_file = f"justification_report_{model_name}.json"
    
    print(f"Generating justification report for {model_name}...")
    data = load_scored_data(file_path)
    report = generate_justification_report(data, model_name)
    
    print_justification_summary(report)
    save_justification_report(report, output_file)

if __name__ == "__main__":
    main() 