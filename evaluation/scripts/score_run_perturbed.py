#!/usr/bin/env python3
"""Enhanced scoring script for perturbed datasets with perturbation metadata."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from score_run import main as score_main
from datasets.fetaqa_perturbed import FetaqaPerturbedDataset
from postprocess.fetaqa_perturbed import FetaqaPerturbedPostProcessor

def analyze_perturbation_performance(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze performance by perturbation class and type."""
    analysis = {
        "by_perturbation_class": {},
        "by_perturbation_type": {},
        "by_modality": {},
        "overall": {}
    }
    
    # Initialize counters
    for record in records:
        pclass = record.get("perturbation_class", "unknown")
        ptype = record.get("perturbation_type", "unknown")
        modality = record.get("modality", "unknown")
        
        # Initialize if not exists
        if pclass not in analysis["by_perturbation_class"]:
            analysis["by_perturbation_class"][pclass] = {"exact_match": 0, "total": 0}
        if ptype not in analysis["by_perturbation_type"]:
            analysis["by_perturbation_type"][ptype] = {"exact_match": 0, "total": 0}
        if modality not in analysis["by_modality"]:
            analysis["by_modality"][modality] = {"exact_match": 0, "total": 0}
    
    # Count results
    for record in records:
        pclass = record.get("perturbation_class", "unknown")
        ptype = record.get("perturbation_type", "unknown")
        modality = record.get("modality", "unknown")
        exact_match = record.get("exact_match", 0)
        
        analysis["by_perturbation_class"][pclass]["total"] += 1
        analysis["by_perturbation_type"][ptype]["total"] += 1
        analysis["by_modality"][modality]["total"] += 1
        
        if exact_match == 1:
            analysis["by_perturbation_class"][pclass]["exact_match"] += 1
            analysis["by_perturbation_type"][ptype]["exact_match"] += 1
            analysis["by_modality"][modality]["exact_match"] += 1
    
    # Calculate percentages
    for category in ["by_perturbation_class", "by_perturbation_type", "by_modality"]:
        for key, data in analysis[category].items():
            if data["total"] > 0:
                data["percentage"] = (data["exact_match"] / data["total"]) * 100
            else:
                data["percentage"] = 0.0
    
    # Overall statistics
    total_exact_matches = sum(1 for r in records if r.get("exact_match") == 1)
    total_records = len(records)
    analysis["overall"] = {
        "total": total_records,
        "exact_match": total_exact_matches,
        "percentage": (total_exact_matches / total_records * 100) if total_records > 0 else 0.0
    }
    
    return analysis

def save_perturbation_analysis(analysis: Dict[str, Any], output_path: Path):
    """Save perturbation analysis to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Score perturbed dataset runs with enhanced analysis")
    parser.add_argument("--run_dir", type=str, required=True,
                       help="Directory containing model outputs")
    parser.add_argument("--dataset", type=str, default="fetaqa_perturbed",
                       help="Dataset type (default: fetaqa_perturbed)")
    parser.add_argument("--perturbation_class", type=str,
                       choices=["invariant", "counterfactual", "unanswerable"],
                       help="Filter by perturbation class")
    parser.add_argument("--perturbation_type", type=str,
                       help="Filter by perturbation type")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for results")
    parser.add_argument("--save_analysis", action="store_true",
                       help="Save detailed perturbation analysis")
    
    args = parser.parse_args()
    
    # Set up dataset with perturbation filters
    dataset_kwargs = {}
    if args.perturbation_class:
        dataset_kwargs["perturbation_class"] = args.perturbation_class
    if args.perturbation_type:
        dataset_kwargs["perturbation_type"] = args.perturbation_type
    
    # Override the dataset class for perturbed data
    if args.dataset == "fetaqa_perturbed":
        from datasets.fetaqa_perturbed import FetaqaPerturbedDataset
        dataset_class = FetaqaPerturbedDataset
    else:
        # Use standard dataset class
        from datasets import DATASETS
        dataset_class = DATASETS.get(args.dataset)
        if not dataset_class:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create dataset instance
    dataset = dataset_class(**dataset_kwargs)
    
    # Load gold and predictions
    print("Loading gold answers...")
    gold = dataset.load_gold()
    print(f"Loaded {len(gold)} gold answers")
    
    print("Loading predictions...")
    pred = dataset.parse_run_dir(args.run_dir)
    print(f"Loaded {len(pred)} predictions")
    
    # Consolidate records
    print("Consolidating records...")
    records = dataset.consolidate(gold, pred)
    print(f"Consolidated {len(records)} records")
    
    # Post-process records
    print("Post-processing records...")
    if args.dataset == "fetaqa_perturbed":
        postprocessor = FetaqaPerturbedPostProcessor()
    else:
        from postprocess import PROCESSORS
        postprocessor_class = PROCESSORS.get(args.dataset)
        if not postprocessor_class:
            raise ValueError(f"No post-processor for dataset: {args.dataset}")
        postprocessor = postprocessor_class()
    
    processed_records = postprocessor.process_records(records)
    print(f"Processed {len(processed_records)} records")
    
    # Score records
    print("Scoring records...")
    from scoring.basic import exact_match, f1_score
    from scoring.cae import cae_score
    from scoring.nlg_metrics import calculate_nlg_metrics
    
    scored_records = []
    total_exact_matches = 0
    total_f1_scores = 0.0
    total_cae_scores = 0.0
    total_rouge_1_scores = 0.0
    total_rouge_2_scores = 0.0
    total_rouge_l_scores = 0.0
    total_bleu_scores = 0.0
    
    for record in processed_records:
        gold_answer = record.get("gold_answer", "")
        pred_answer = record.get("pred_answer", "")
        
        # Calculate scores
        em_score = exact_match(gold_answer, pred_answer)
        f1 = f1_score(gold_answer, pred_answer)
        cae = cae_score(gold_answer, pred_answer)
        nlg_metrics = calculate_nlg_metrics(gold_answer, pred_answer)
        
        # Add scores to record
        record["exact_match"] = em_score
        record["f1_score"] = f1
        record["cae_score"] = cae
        record["rouge_1"] = nlg_metrics["rouge_1"]
        record["rouge_2"] = nlg_metrics["rouge_2"]
        record["rouge_l"] = nlg_metrics["rouge_l"]
        record["bleu"] = nlg_metrics["bleu"]
        
        scored_records.append(record)
        
        # Accumulate totals
        total_exact_matches += em_score
        total_f1_scores += f1
        total_cae_scores += cae
        total_rouge_1_scores += nlg_metrics["rouge_1"]
        total_rouge_2_scores += nlg_metrics["rouge_2"]
        total_rouge_l_scores += nlg_metrics["rouge_l"]
        total_bleu_scores += nlg_metrics["bleu"]
    
    # Calculate averages
    num_records = len(scored_records)
    avg_scores = {
        "exact_match": (total_exact_matches / num_records * 100) if num_records > 0 else 0.0,
        "f1_score": (total_f1_scores / num_records * 100) if num_records > 0 else 0.0,
        "cae_score": (total_cae_scores / num_records * 100) if num_records > 0 else 0.0,
        "rouge_1": (total_rouge_1_scores / num_records * 100) if num_records > 0 else 0.0,
        "rouge_2": (total_rouge_2_scores / num_records * 100) if num_records > 0 else 0.0,
        "rouge_l": (total_rouge_l_scores / num_records * 100) if num_records > 0 else 0.0,
        "bleu": (total_bleu_scores / num_records * 100) if num_records > 0 else 0.0,
        "total_records": num_records
    }
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scored records
    scored_file = output_dir / "scored_perturbed.json"
    with open(scored_file, 'w', encoding='utf-8') as f:
        json.dump(scored_records, f, indent=2, ensure_ascii=False)
    print(f"Saved scored records to: {scored_file}")
    
    # Save average scores
    avg_file = output_dir / "avg_scores_perturbed.json"
    with open(avg_file, 'w', encoding='utf-8') as f:
        json.dump(avg_scores, f, indent=2, ensure_ascii=False)
    print(f"Saved average scores to: {avg_file}")
    
    # Generate and save perturbation analysis
    if args.save_analysis:
        analysis = analyze_perturbation_performance(scored_records)
        analysis_file = output_dir / "perturbation_analysis.json"
        save_perturbation_analysis(analysis, analysis_file)
        print(f"Saved perturbation analysis to: {analysis_file}")
        
        # Print summary
        print("\n=== Perturbation Analysis Summary ===")
        print(f"Overall Exact Match: {avg_scores['exact_match']:.2f}%")
        print(f"Overall F1 Score: {avg_scores['f1_score']:.2f}%")
        print(f"Overall CAE Score: {avg_scores['cae_score']:.2f}%")
        print(f"Overall ROUGE-1 Score: {avg_scores['rouge_1']:.2f}%")
        print(f"Overall ROUGE-2 Score: {avg_scores['rouge_2']:.2f}%")
        print(f"Overall ROUGE-L Score: {avg_scores['rouge_l']:.2f}%")
        print(f"Overall BLEU Score: {avg_scores['bleu']:.2f}%")
        
        print("\nBy Perturbation Class:")
        for pclass, data in analysis["by_perturbation_class"].items():
            print(f"  {pclass}: {data['percentage']:.2f}% ({data['exact_match']}/{data['total']})")
        
        print("\nBy Perturbation Type:")
        for ptype, data in analysis["by_perturbation_type"].items():
            print(f"  {ptype}: {data['percentage']:.2f}% ({data['exact_match']}/{data['total']})")
        
        print("\nBy Modality:")
        for modality, data in analysis["by_modality"].items():
            print(f"  {modality}: {data['percentage']:.2f}% ({data['exact_match']}/{data['total']})")
    
    print(f"\nScoring complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 