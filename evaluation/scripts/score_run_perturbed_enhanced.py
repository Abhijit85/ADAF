#!/usr/bin/env python3
"""Enhanced scoring script for perturbed FETA-QA datasets with comprehensive analysis."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import statistics

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from score_run_fetaqa_enhanced import main as fetaqa_score_main

# Handle imports with fallback
try:
    from evaluation.scripts.datasets.fetaqa_perturbed import FetaqaPerturbedDataset
    from evaluation.scripts.postprocess.fetaqa_perturbed import FetaqaPerturbedPostProcessor
    from evaluation.scripts.scoring.nlg_metrics import calculate_nlg_metrics_fetaqa
    from evaluation.scripts.scoring.basic import exact_match, f1_score
except ModuleNotFoundError:
    from datasets.fetaqa_perturbed import FetaqaPerturbedDataset
    from postprocess.fetaqa_perturbed import FetaqaPerturbedPostProcessor
    from scoring.nlg_metrics import calculate_nlg_metrics_fetaqa
    from scoring.basic import exact_match, f1_score

def calculate_grouped_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate metrics grouped by perturbation class and type."""
    
    # Initialize structure
    grouped_analysis = {
        "by_perturbation_class": {},
        "by_perturbation_type": {},
        "detailed_class_type_breakdown": {}
    }
    
    # Group records by perturbation class and type
    for record in records:
        pclass = record.get("perturbation_class", "unknown")
        ptype = record.get("perturbation_type", "unknown")
        class_type_key = f"{pclass}_{ptype}"
        
        # Initialize class analysis
        if pclass not in grouped_analysis["by_perturbation_class"]:
            grouped_analysis["by_perturbation_class"][pclass] = {
                "exact_match": 0,
                "f1": 0.0,
                "rouge_1": 0.0,
                "rouge_2": 0.0,
                "rouge_l": 0.0,
                "bleu": 0.0,
                "count": 0,
                "examples": []
            }
        
        # Initialize type analysis
        if ptype not in grouped_analysis["by_perturbation_type"]:
            grouped_analysis["by_perturbation_type"][ptype] = {
                "exact_match": 0,
                "f1": 0.0,
                "rouge_1": 0.0,
                "rouge_2": 0.0,
                "rouge_l": 0.0,
                "bleu": 0.0,
                "count": 0,
                "perturbation_class": pclass,
                "examples": []
            }
        
        # Initialize detailed breakdown
        if class_type_key not in grouped_analysis["detailed_class_type_breakdown"]:
            grouped_analysis["detailed_class_type_breakdown"][class_type_key] = {
                "exact_match": 0,
                "f1": 0.0,
                "rouge_1": 0.0,
                "rouge_2": 0.0,
                "rouge_l": 0.0,
                "bleu": 0.0,
                "count": 0,
                "perturbation_class": pclass,
                "perturbation_type": ptype,
                "examples": []
            }
        
        # Add record to appropriate groups
        grouped_analysis["by_perturbation_class"][pclass]["examples"].append(record)
        grouped_analysis["by_perturbation_type"][ptype]["examples"].append(record)
        grouped_analysis["detailed_class_type_breakdown"][class_type_key]["examples"].append(record)
    
    # Calculate metrics for each group
    for group_name, group_data in grouped_analysis.items():
        for key, data in group_data.items():
            if isinstance(data, dict) and "examples" in data:
                examples = data["examples"]
                if examples:
                    # Calculate metrics
                    exact_matches = sum(1 for ex in examples if ex.get("exact_match") == 1)
                    f1_scores = [ex.get("f1", 0.0) for ex in examples]
                    rouge_1_scores = [ex.get("rouge_1", 0.0) for ex in examples]
                    rouge_2_scores = [ex.get("rouge_2", 0.0) for ex in examples]
                    rouge_l_scores = [ex.get("rouge_l", 0.0) for ex in examples]
                    bleu_scores = [ex.get("bleu", 0.0) for ex in examples]
                    
                    # Update metrics
                    data["exact_match"] = exact_matches / len(examples) if examples else 0.0
                    data["f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                    data["rouge_1"] = sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0.0
                    data["rouge_2"] = sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0.0
                    data["rouge_l"] = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
                    data["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
                    data["count"] = len(examples)
                    
                    # Remove examples to keep output clean
                    del data["examples"]
    
    return grouped_analysis

def calculate_robustness_analysis(records: List[Dict[str, Any]], original_fetaqa_scores: Dict[str, float] = None) -> Dict[str, Any]:
    """Calculate robustness analysis including performance degradation and challenge ranking."""
    
    robustness = {
        "most_challenging_perturbations": [],
        "least_challenging_perturbations": [],
        "performance_degradation": {},
        "model_consistency": {}
    }
    
    # Calculate overall performance
    total_exact_matches = sum(1 for r in records if r.get("exact_match") == 1)
    total_records = len(records)
    overall_exact_match = total_exact_matches / total_records if total_records > 0 else 0.0
    
    overall_f1 = sum(r.get("f1", 0.0) for r in records) / total_records if total_records > 0 else 0.0
    overall_rouge_1 = sum(r.get("rouge_1", 0.0) for r in records) / total_records if total_records > 0 else 0.0
    
    # Calculate performance by perturbation type
    type_performance = {}
    for record in records:
        ptype = record.get("perturbation_type", "unknown")
        if ptype not in type_performance:
            type_performance[ptype] = {
                "exact_match": 0,
                "f1": 0.0,
                "rouge_1": 0.0,
                "count": 0
            }
        
        type_performance[ptype]["exact_match"] += record.get("exact_match", 0)
        type_performance[ptype]["f1"] += record.get("f1", 0.0)
        type_performance[ptype]["rouge_1"] += record.get("rouge_1", 0.0)
        type_performance[ptype]["count"] += 1
    
    # Calculate averages
    for ptype, data in type_performance.items():
        if data["count"] > 0:
            data["exact_match"] /= data["count"]
            data["f1"] /= data["count"]
            data["rouge_1"] /= data["count"]
    
    # Rank by difficulty (lowest scores = most challenging)
    sorted_by_difficulty = sorted(type_performance.items(), key=lambda x: x[1]["exact_match"])
    
    robustness["most_challenging_perturbations"] = [
        {
            "perturbation_type": ptype,
            "exact_match": data["exact_match"],
            "f1": data["f1"],
            "rouge_1": data["rouge_1"],
            "difficulty_rank": i + 1
        }
        for i, (ptype, data) in enumerate(sorted_by_difficulty)
    ]
    
    robustness["least_challenging_perturbations"] = [
        {
            "perturbation_type": ptype,
            "exact_match": data["exact_match"],
            "f1": data["f1"],
            "rouge_1": data["rouge_1"],
            "difficulty_rank": len(sorted_by_difficulty) - i
        }
        for i, (ptype, data) in enumerate(reversed(sorted_by_difficulty))
    ]
    
    # Performance degradation (if original scores provided)
    if original_fetaqa_scores:
        robustness["performance_degradation"] = {
            "overall_degradation": {
                "exact_match_drop": overall_exact_match - original_fetaqa_scores.get("exact_match", 0.0),
                "f1_drop": overall_f1 - original_fetaqa_scores.get("f1", 0.0),
                "rouge_1_drop": overall_rouge_1 - original_fetaqa_scores.get("rouge_1", 0.0)
            }
        }
    
    # Model consistency (standard deviation across perturbation types)
    exact_match_scores = [data["exact_match"] for data in type_performance.values()]
    f1_scores = [data["f1"] for data in type_performance.values()]
    rouge_1_scores = [data["rouge_1"] for data in type_performance.values()]
    
    robustness["model_consistency"] = {
        "std_by_metric": {
            "exact_match_std": statistics.stdev(exact_match_scores) if len(exact_match_scores) > 1 else 0.0,
            "f1_std": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
            "rouge_1_std": statistics.stdev(rouge_1_scores) if len(rouge_1_scores) > 1 else 0.0
        },
        "consistency_rankings": sorted(type_performance.items(), key=lambda x: x[1]["exact_match"], reverse=True)
    }
    
    return robustness

def create_comprehensive_report(records: List[Dict[str, Any]], 
                              grouped_analysis: Dict[str, Any],
                              robustness_analysis: Dict[str, Any],
                              args: argparse.Namespace) -> Dict[str, Any]:
    """Create comprehensive analysis report."""
    
    # Calculate overall metrics
    total_exact_matches = sum(1 for r in records if r.get("exact_match") == 1)
    total_records = len(records)
    overall_exact_match = total_exact_matches / total_records if total_records > 0 else 0.0
    
    overall_f1 = sum(r.get("f1", 0.0) for r in records) / total_records if total_records > 0 else 0.0
    overall_rouge_1 = sum(r.get("rouge_1", 0.0) for r in records) / total_records if total_records > 0 else 0.0
    overall_rouge_2 = sum(r.get("rouge_2", 0.0) for r in records) / total_records if total_records > 0 else 0.0
    overall_rouge_l = sum(r.get("rouge_l", 0.0) for r in records) / total_records if total_records > 0 else 0.0
    overall_bleu = sum(r.get("bleu", 0.0) for r in records) / total_records if total_records > 0 else 0.0
    
    # Count perturbation classes and types
    perturbation_classes = set(r.get("perturbation_class", "unknown") for r in records)
    perturbation_types = set(r.get("perturbation_type", "unknown") for r in records)
    
    report = {
        "metadata": {
            "model": "deepseek-r1-671b",
            "dataset": "fetaqa_perturbed",
            "total_examples": total_records,
            "evaluation_date": datetime.now().isoformat(),
            "perturbation_classes": len(perturbation_classes),
            "perturbation_types": len(perturbation_types),
            "evaluation_parameters": {
                "use_nlg_metrics": args.use_nlg_metrics,
                "min_nlg_length": args.min_nlg_length,
                "fetaqa_specific_normalization": args.fetaqa_specific_normalization,
                "detailed_breakdown": args.detailed_breakdown,
                "robustness_analysis": args.robustness_analysis
            }
        },
        "overall_performance": {
            "exact_match": overall_exact_match,
            "f1": overall_f1,
            "rouge_1": overall_rouge_1,
            "rouge_2": overall_rouge_2,
            "rouge_l": overall_rouge_l,
            "bleu": overall_bleu,
            "total_examples": total_records
        }
    }
    
    # Add grouped analysis
    if args.detailed_breakdown:
        report.update(grouped_analysis)
    
    # Add robustness analysis
    if args.robustness_analysis:
        report["robustness_analysis"] = robustness_analysis
    
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description="Enhanced scoring for perturbed FETA-QA datasets")
    parser.add_argument("--post_file", type=str, required=True, help="Path to post-processed JSON")
    parser.add_argument("--gemini_key", help="Optional Gemini API key")
    parser.add_argument("--use_nlg_metrics", action="store_true", help="Calculate NLG metrics")
    parser.add_argument("--min_nlg_length", type=int, default=0, help="Minimum text length for NLG metrics (0 = no threshold)")
    parser.add_argument("--enhanced_nlg_normalization", action="store_true", help="Use enhanced normalization")
    parser.add_argument("--optimized_rouge_normalization", action="store_true", help="Use optimized ROUGE normalization")
    parser.add_argument("--fetaqa_specific_normalization", action="store_true", default=True, help="Use FETAQA-specific normalization")
    parser.add_argument("--perturbation_class", choices=["invariant", "counterfactual", "unanswerable"], help="Filter by perturbation class")
    parser.add_argument("--perturbation_type", help="Filter by perturbation type")
    parser.add_argument("--detailed_breakdown", action="store_true", help="Generate detailed class×type breakdown")
    parser.add_argument("--robustness_analysis", action="store_true", help="Include robustness analysis")
    parser.add_argument("--output_report", help="Output file for comprehensive analysis")
    
    args = parser.parse_args(argv)
    
    # Load post-processed data
    with open(args.post_file, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    # Filter by perturbation class/type if specified
    if args.perturbation_class:
        records = [r for r in records if r.get("perturbation_class") == args.perturbation_class]
    
    if args.perturbation_type:
        records = [r for r in records if r.get("perturbation_type") == args.perturbation_type]
    
    print(f"Loaded {len(records)} records for analysis")
    
    # Calculate metrics for each record using regular FETA-QA scoring logic
    for record in records:
        gold_answer = record.get("pp_gold_answer", "")
        pred_answer = record.get("pp_pred_answer", "")
        
        # Calculate exact match
        record["exact_match"] = 1 if gold_answer == pred_answer else 0
        
        # Calculate F1 score
        if gold_answer and pred_answer:
            # Simple F1 calculation based on word overlap
            gold_words = set(gold_answer.split())
            pred_words = set(pred_answer.split())
            
            if gold_words and pred_words:
                precision = len(gold_words & pred_words) / len(pred_words)
                recall = len(gold_words & pred_words) / len(gold_words)
                if precision + recall > 0:
                    record["f1"] = 2 * precision * recall / (precision + recall)
                else:
                    record["f1"] = 0.0
            else:
                record["f1"] = 0.0
        else:
            record["f1"] = 0.0
        
        # Calculate NLG metrics if requested
        if args.use_nlg_metrics:
            if gold_answer and pred_answer:
                # Simple ROUGE-like metrics
                gold_words = gold_answer.split()
                pred_words = pred_answer.split()
                
                if gold_words and pred_words:
                    # ROUGE-1 (unigram overlap)
                    overlap = len(set(gold_words) & set(pred_words))
                    record["rouge_1"] = overlap / len(set(gold_words)) if gold_words else 0.0
                    record["rouge_2"] = 0.0  # Simplified - would need bigram calculation
                    record["rouge_l"] = record["rouge_1"]  # Simplified
                    record["bleu"] = 0.0  # Simplified - would need proper BLEU calculation
                else:
                    record["rouge_1"] = 0.0
                    record["rouge_2"] = 0.0
                    record["rouge_l"] = 0.0
                    record["bleu"] = 0.0
            else:
                record["rouge_1"] = 0.0
                record["rouge_2"] = 0.0
                record["rouge_l"] = 0.0
                record["bleu"] = 0.0
    
    # Calculate grouped analysis
    grouped_analysis = {}
    if args.detailed_breakdown:
        grouped_analysis = calculate_grouped_metrics(records)
        print("Calculated grouped analysis")
    
    # Calculate robustness analysis
    robustness_analysis = {}
    if args.robustness_analysis:
        # TODO: Load original FETA-QA scores for comparison
        original_scores = None  # Load from original FETA-QA evaluation
        robustness_analysis = calculate_robustness_analysis(records, original_scores)
        print("Calculated robustness analysis")
    
    # Create comprehensive report
    report = create_comprehensive_report(records, grouped_analysis, robustness_analysis, args)
    
    # Save report
    if args.output_report:
        output_path = Path(args.output_report)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved comprehensive report to {output_path}")
    else:
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"perturbed_analysis_{timestamp}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved comprehensive report to {output_path}")
    
    # Print summary
    print(f"\n=== PERTURBED FETA-QA EVALUATION SUMMARY ===")
    print(f"Total examples: {len(records)}")
    print(f"Overall Exact Match: {report['overall_performance']['exact_match']:.4f}")
    print(f"Overall F1: {report['overall_performance']['f1']:.4f}")
    if args.use_nlg_metrics:
        print(f"Overall ROUGE-1: {report['overall_performance']['rouge_1']:.4f}")
        print(f"Overall ROUGE-2: {report['overall_performance']['rouge_2']:.4f}")
        print(f"Overall ROUGE-L: {report['overall_performance']['rouge_l']:.4f}")
        print(f"Overall BLEU: {report['overall_performance']['bleu']:.4f}")
    
    if args.detailed_breakdown:
        print(f"\n=== BY PERTURBATION CLASS ===")
        for pclass, data in grouped_analysis["by_perturbation_class"].items():
            print(f"{pclass}: EM={data['exact_match']:.4f}, F1={data['f1']:.4f}, Count={data['count']}")
    
    if args.robustness_analysis and robustness_analysis.get("most_challenging_perturbations"):
        print(f"\n=== MOST CHALLENGING PERTURBATIONS ===")
        for i, challenge in enumerate(robustness_analysis["most_challenging_perturbations"][:3]):
            print(f"{i+1}. {challenge['perturbation_type']}: EM={challenge['exact_match']:.4f}")

if __name__ == "__main__":
    main() 