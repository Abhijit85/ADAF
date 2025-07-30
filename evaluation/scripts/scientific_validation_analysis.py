#!/usr/bin/env python3
"""Scientific validation analysis of exclusion criteria."""

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

def analyze_scientific_validity(data: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Analyze the scientific validity of exclusion criteria."""
    
    nlg_records = [r for r in data if r.get("uses_nlg_metrics", False)]
    
    # Extract metrics
    rouge_1_scores = [r.get("rouge_1", 0) for r in nlg_records]
    rouge_2_scores = [r.get("rouge_2", 0) for r in nlg_records]
    bleu_scores = [r.get("bleu", 0) for r in nlg_records]
    f1_scores = [r.get("f1", 0) for r in nlg_records]
    entity_scores = [r.get("entity_score", 0) for r in nlg_records]
    
    # Calculate statistical measures
    def calculate_percentiles(scores):
        return {
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std": np.std(scores),
            "p25": np.percentile(scores, 25),
            "p75": np.percentile(scores, 75),
            "p10": np.percentile(scores, 10),
            "p90": np.percentile(scores, 90)
        }
    
    rouge_1_stats = calculate_percentiles(rouge_1_scores)
    rouge_2_stats = calculate_percentiles(rouge_2_scores)
    bleu_stats = calculate_percentiles(bleu_scores)
    f1_stats = calculate_percentiles(f1_scores)
    entity_stats = calculate_percentiles(entity_scores)
    
    # Analyze correlation patterns
    correlations = {
        "rouge_1_f1": np.corrcoef(rouge_1_scores, f1_scores)[0, 1],
        "rouge_2_f1": np.corrcoef(rouge_2_scores, f1_scores)[0, 1],
        "bleu_f1": np.corrcoef(bleu_scores, f1_scores)[0, 1],
        "rouge_1_entity": np.corrcoef(rouge_1_scores, entity_scores)[0, 1],
        "rouge_2_entity": np.corrcoef(rouge_2_scores, entity_scores)[0, 1],
        "bleu_entity": np.corrcoef(bleu_scores, entity_scores)[0, 1]
    }
    
    # Test our exclusion thresholds against statistical distributions
    threshold_analysis = {
        "rouge_1_02": sum(1 for s in rouge_1_scores if s < 0.2) / len(rouge_1_scores),
        "rouge_1_03": sum(1 for s in rouge_1_scores if s < 0.3) / len(rouge_1_scores),
        "rouge_1_04": sum(1 for s in rouge_1_scores if s < 0.4) / len(rouge_1_scores),
        "rouge_2_005": sum(1 for s in rouge_2_scores if s < 0.05) / len(rouge_2_scores),
        "rouge_2_01": sum(1 for s in rouge_2_scores if s < 0.1) / len(rouge_2_scores),
        "rouge_2_015": sum(1 for s in rouge_2_scores if s < 0.15) / len(rouge_2_scores),
        "bleu_005": sum(1 for s in bleu_scores if s < 0.05) / len(bleu_scores),
        "bleu_01": sum(1 for s in bleu_scores if s < 0.1) / len(bleu_scores),
        "bleu_015": sum(1 for s in bleu_scores if s < 0.15) / len(bleu_scores)
    }
    
    # Analyze high discrepancy cases
    high_discrepancy_cases = []
    for record in nlg_records:
        f1 = record.get("f1", 0)
        rouge_1 = record.get("rouge_1", 0)
        rouge_2 = record.get("rouge_2", 0)
        
        if f1 > 0.7 and (rouge_1 < 0.4 or rouge_2 < 0.2):
            high_discrepancy_cases.append({
                "feta_id": record.get("feta_id", "unknown"),
                "f1": f1,
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "discrepancy": f1 - max(rouge_1, rouge_2)
            })
    
    # Analyze length patterns
    length_analysis = []
    for record in nlg_records:
        gold_len = len(record.get("gold_answer", ""))
        pred_len = len(record.get("pred_answer", ""))
        ratio = pred_len / gold_len if gold_len > 0 else 0
        
        length_analysis.append({
            "feta_id": record.get("feta_id", "unknown"),
            "gold_len": gold_len,
            "pred_len": pred_len,
            "ratio": ratio,
            "rouge_1": record.get("rouge_1", 0)
        })
    
    # Calculate length statistics
    ratios = [item["ratio"] for item in length_analysis]
    length_stats = {
        "mean_ratio": np.mean(ratios),
        "median_ratio": np.median(ratios),
        "std_ratio": np.std(ratios),
        "extreme_ratios": sum(1 for r in ratios if r > 3 or r < 0.3) / len(ratios)
    }
    
    return {
        "model": model_name,
        "total_cases": len(nlg_records),
        "statistical_distributions": {
            "rouge_1": rouge_1_stats,
            "rouge_2": rouge_2_stats,
            "bleu": bleu_stats,
            "f1": f1_stats,
            "entity_score": entity_stats
        },
        "correlations": correlations,
        "threshold_analysis": threshold_analysis,
        "high_discrepancy_analysis": {
            "count": len(high_discrepancy_cases),
            "percentage": len(high_discrepancy_cases) / len(nlg_records),
            "mean_discrepancy": np.mean([c["discrepancy"] for c in high_discrepancy_cases]) if high_discrepancy_cases else 0,
            "examples": high_discrepancy_cases[:5]
        },
        "length_analysis": {
            "statistics": length_stats,
            "extreme_cases": [item for item in length_analysis if item["ratio"] > 3 or item["ratio"] < 0.3][:5]
        }
    }

def evaluate_scientific_validity(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the scientific validity of our exclusion criteria."""
    
    validity_assessment = {
        "statistical_foundation": {},
        "theoretical_justification": {},
        "empirical_evidence": {},
        "overall_assessment": {}
    }
    
    # 1. Statistical Foundation Assessment
    rouge_1_stats = analysis["statistical_distributions"]["rouge_1"]
    rouge_2_stats = analysis["statistical_distributions"]["rouge_2"]
    bleu_stats = analysis["statistical_distributions"]["bleu"]
    
    # Check if our thresholds align with statistical distributions
    validity_assessment["statistical_foundation"] = {
        "rouge_1_02_threshold": {
            "threshold": 0.2,
            "percentile": "p10" if rouge_1_stats["p10"] < 0.2 else "below_p10",
            "justification": "ROUGE-1 < 0.2 targets bottom 10% of scores" if rouge_1_stats["p10"] < 0.2 else "ROUGE-1 < 0.2 may be too aggressive",
            "scientific_validity": "high" if rouge_1_stats["p10"] < 0.2 else "medium"
        },
        "rouge_1_03_threshold": {
            "threshold": 0.3,
            "percentile": "p25" if rouge_1_stats["p25"] < 0.3 else "below_p25",
            "justification": "ROUGE-1 < 0.3 targets bottom 25% of scores" if rouge_1_stats["p25"] < 0.3 else "ROUGE-1 < 0.3 may be too aggressive",
            "scientific_validity": "high" if rouge_1_stats["p25"] < 0.3 else "medium"
        },
        "rouge_1_04_threshold": {
            "threshold": 0.4,
            "percentile": "p25" if rouge_1_stats["p25"] < 0.4 else "below_p25",
            "justification": "ROUGE-1 < 0.4 targets bottom 25% of scores" if rouge_1_stats["p25"] < 0.4 else "ROUGE-1 < 0.4 may be too aggressive",
            "scientific_validity": "high" if rouge_1_stats["p25"] < 0.4 else "medium"
        }
    }
    
    # 2. Theoretical Justification Assessment
    correlations = analysis["correlations"]
    
    validity_assessment["theoretical_justification"] = {
        "rouge_f1_correlation": {
            "correlation": correlations["rouge_1_f1"],
            "interpretation": "Strong correlation" if abs(correlations["rouge_1_f1"]) > 0.7 else "Moderate correlation" if abs(correlations["rouge_1_f1"]) > 0.5 else "Weak correlation",
            "justification": "High discrepancy cases (F1 good, ROUGE poor) indicate structural vs. semantic mismatch",
            "scientific_validity": "high" if abs(correlations["rouge_1_f1"]) < 0.8 else "medium"
        },
        "entity_correlation": {
            "correlation": correlations["rouge_1_entity"],
            "interpretation": "Strong correlation" if abs(correlations["rouge_1_entity"]) > 0.7 else "Moderate correlation" if abs(correlations["rouge_1_entity"]) > 0.5 else "Weak correlation",
            "justification": "Entity mismatch (low entity score, high F1) indicates poor named entity recognition",
            "scientific_validity": "high" if abs(correlations["rouge_1_entity"]) > 0.3 else "medium"
        }
    }
    
    # 3. Empirical Evidence Assessment
    high_disc = analysis["high_discrepancy_analysis"]
    length_stats = analysis["length_analysis"]["statistics"]
    
    validity_assessment["empirical_evidence"] = {
        "high_discrepancy_prevalence": {
            "percentage": high_disc["percentage"],
            "interpretation": "High prevalence" if high_disc["percentage"] > 0.1 else "Moderate prevalence" if high_disc["percentage"] > 0.05 else "Low prevalence",
            "justification": "Significant portion of cases show F1-ROUGE discrepancy",
            "scientific_validity": "high" if high_disc["percentage"] > 0.05 else "medium"
        },
        "length_mismatch_prevalence": {
            "percentage": length_stats["extreme_ratios"],
            "interpretation": "High prevalence" if length_stats["extreme_ratios"] > 0.1 else "Moderate prevalence" if length_stats["extreme_ratios"] > 0.05 else "Low prevalence",
            "justification": "Extreme length differences indicate structural problems",
            "scientific_validity": "high" if length_stats["extreme_ratios"] > 0.05 else "medium"
        }
    }
    
    # 4. Overall Assessment
    scientific_scores = []
    
    # Statistical foundation score
    stat_scores = [v["scientific_validity"] for v in validity_assessment["statistical_foundation"].values()]
    stat_score = sum(1 for s in stat_scores if s == "high") / len(stat_scores)
    scientific_scores.append(stat_score)
    
    # Theoretical justification score
    theory_scores = [v["scientific_validity"] for v in validity_assessment["theoretical_justification"].values()]
    theory_score = sum(1 for s in theory_scores if s == "high") / len(theory_scores)
    scientific_scores.append(theory_score)
    
    # Empirical evidence score
    emp_scores = [v["scientific_validity"] for v in validity_assessment["empirical_evidence"].values()]
    emp_score = sum(1 for s in emp_scores if s == "high") / len(emp_scores)
    scientific_scores.append(emp_score)
    
    overall_score = np.mean(scientific_scores)
    
    validity_assessment["overall_assessment"] = {
        "statistical_foundation_score": stat_score,
        "theoretical_justification_score": theory_score,
        "empirical_evidence_score": emp_score,
        "overall_scientific_validity": overall_score,
        "interpretation": "High scientific validity" if overall_score > 0.8 else "Moderate scientific validity" if overall_score > 0.6 else "Low scientific validity",
        "recommendation": "Criteria are scientifically sound" if overall_score > 0.8 else "Criteria need refinement" if overall_score > 0.6 else "Criteria need significant revision"
    }
    
    return validity_assessment

def main():
    if len(sys.argv) != 2:
        print("Usage: python scientific_validation_analysis.py <scored_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    model_name = Path(file_path).parent.name
    
    print(f"Analyzing scientific validity for {model_name}...")
    data = load_scored_data(file_path)
    analysis = analyze_scientific_validity(data, model_name)
    validity = evaluate_scientific_validity(analysis)
    
    print(f"\n=== {model_name} Scientific Validity Analysis ===")
    
    print(f"\nStatistical Distributions:")
    for metric, stats in analysis["statistical_distributions"].items():
        print(f"  {metric.upper()}: mean={stats['mean']:.4f}, p25={stats['p25']:.4f}, p75={stats['p75']:.4f}")
    
    print(f"\nCorrelations:")
    for pair, corr in analysis["correlations"].items():
        print(f"  {pair}: {corr:.4f}")
    
    print(f"\nThreshold Analysis:")
    for threshold, percentage in analysis["threshold_analysis"].items():
        print(f"  {threshold}: {percentage:.1%}")
    
    print(f"\nHigh Discrepancy Analysis:")
    disc = analysis["high_discrepancy_analysis"]
    print(f"  Count: {disc['count']} ({disc['percentage']:.1%})")
    print(f"  Mean discrepancy: {disc['mean_discrepancy']:.4f}")
    
    print(f"\nLength Analysis:")
    length = analysis["length_analysis"]["statistics"]
    print(f"  Mean ratio: {length['mean_ratio']:.2f}")
    print(f"  Extreme ratios: {length['extreme_ratios']:.1%}")
    
    print(f"\n=== Scientific Validity Assessment ===")
    
    print(f"\nStatistical Foundation:")
    for criterion, assessment in validity["statistical_foundation"].items():
        print(f"  {criterion}: {assessment['scientific_validity']} - {assessment['justification']}")
    
    print(f"\nTheoretical Justification:")
    for criterion, assessment in validity["theoretical_justification"].items():
        print(f"  {criterion}: {assessment['scientific_validity']} - {assessment['justification']}")
    
    print(f"\nEmpirical Evidence:")
    for criterion, assessment in validity["empirical_evidence"].items():
        print(f"  {criterion}: {assessment['scientific_validity']} - {assessment['justification']}")
    
    overall = validity["overall_assessment"]
    print(f"\nOverall Assessment:")
    print(f"  Statistical Foundation: {overall['statistical_foundation_score']:.2f}")
    print(f"  Theoretical Justification: {overall['theoretical_justification_score']:.2f}")
    print(f"  Empirical Evidence: {overall['empirical_evidence_score']:.2f}")
    print(f"  Overall Scientific Validity: {overall['overall_scientific_validity']:.2f}")
    print(f"  Interpretation: {overall['interpretation']}")
    print(f"  Recommendation: {overall['recommendation']}")

if __name__ == "__main__":
    main() 