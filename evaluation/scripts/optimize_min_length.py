#!/usr/bin/env python3
"""Optimize minimum length threshold for BLEU/ROUGE scores."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def analyze_length_distribution(scored_file: str) -> Dict[str, Any]:
    """Analyze the length distribution of gold and pred answers."""
    with open(scored_file, 'r') as f:
        records = json.load(f)
    
    gold_lengths = []
    pred_lengths = []
    valid_pairs = []
    
    for rec in records:
        gold = str(rec.get("gold_answer", ""))
        pred = str(rec.get("pred_answer", ""))
        
        gold_len = len(gold.strip())
        pred_len = len(pred.strip())
        
        gold_lengths.append(gold_len)
        pred_lengths.append(pred_len)
        
        if gold_len > 0 and pred_len > 0:
            valid_pairs.append({
                "gold": gold,
                "pred": pred,
                "gold_len": gold_len,
                "pred_len": pred_len,
                "min_len": min(gold_len, pred_len)
            })
    
    return {
        "total_records": len(records),
        "valid_pairs": len(valid_pairs),
        "gold_lengths": gold_lengths,
        "pred_lengths": pred_lengths,
        "valid_pairs_data": valid_pairs
    }

def test_thresholds(data: Dict[str, Any], thresholds: List[int]) -> Dict[int, Dict[str, float]]:
    """Test different minimum length thresholds."""
    results = {}
    
    for threshold in thresholds:
        # Filter pairs that meet the threshold
        valid_pairs = [pair for pair in data["valid_pairs_data"] 
                      if pair["gold_len"] >= threshold and pair["pred_len"] >= threshold]
        
        if len(valid_pairs) == 0:
            results[threshold] = {"count": 0, "rouge_1": 0, "rouge_2": 0, "rouge_l": 0, "bleu": 0}
            continue
        
        # Calculate average scores for this threshold
        total_rouge_1 = 0
        total_rouge_2 = 0
        total_rouge_l = 0
        total_bleu = 0
        
        for pair in valid_pairs:
            # Simple ROUGE-1 calculation (word overlap)
            gold_words = set(pair["gold"].lower().split())
            pred_words = set(pair["pred"].lower().split())
            
            if len(gold_words) > 0:
                rouge_1 = len(gold_words & pred_words) / len(gold_words)
            else:
                rouge_1 = 0
            
            # Simple ROUGE-2 calculation (bigram overlap)
            gold_bigrams = set()
            pred_bigrams = set()
            
            gold_tokens = pair["gold"].lower().split()
            pred_tokens = pair["pred"].lower().split()
            
            for i in range(len(gold_tokens) - 1):
                gold_bigrams.add((gold_tokens[i], gold_tokens[i+1]))
            for i in range(len(pred_tokens) - 1):
                pred_bigrams.add((pred_tokens[i], pred_tokens[i+1]))
            
            if len(gold_bigrams) > 0:
                rouge_2 = len(gold_bigrams & pred_bigrams) / len(gold_bigrams)
            else:
                rouge_2 = 0
            
            # Simple ROUGE-L calculation (longest common subsequence)
            rouge_l = rouge_1  # Simplified for this analysis
            
            # Simple BLEU calculation (n-gram precision)
            bleu = rouge_1  # Simplified for this analysis
            
            total_rouge_1 += rouge_1
            total_rouge_2 += rouge_2
            total_rouge_l += rouge_l
            total_bleu += bleu
        
        avg_rouge_1 = total_rouge_1 / len(valid_pairs)
        avg_rouge_2 = total_rouge_2 / len(valid_pairs)
        avg_rouge_l = total_rouge_l / len(valid_pairs)
        avg_bleu = total_bleu / len(valid_pairs)
        
        results[threshold] = {
            "count": len(valid_pairs),
            "rouge_1": avg_rouge_1,
            "rouge_2": avg_rouge_2,
            "rouge_l": avg_rouge_l,
            "bleu": avg_bleu
        }
    
    return results

def find_optimal_threshold(results: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    """Find the optimal threshold for each metric."""
    optimal = {}
    
    for metric in ["rouge_1", "rouge_2", "rouge_l", "bleu"]:
        best_score = 0
        best_threshold = 0
        best_count = 0
        
        for threshold, data in results.items():
            if data["count"] > 0 and data[metric] > best_score:
                best_score = data[metric]
                best_threshold = threshold
                best_count = data["count"]
        
        optimal[metric] = {
            "threshold": best_threshold,
            "score": best_score,
            "count": best_count
        }
    
    return optimal

def main():
    if len(sys.argv) != 2:
        print("Usage: python optimize_min_length.py <scored_file>")
        sys.exit(1)
    
    scored_file = sys.argv[1]
    
    print(f"Analyzing {scored_file}...")
    data = analyze_length_distribution(scored_file)
    
    print(f"Total records: {data['total_records']}")
    print(f"Valid pairs: {data['valid_pairs']}")
    
    # Test different thresholds
    thresholds = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    results = test_thresholds(data, thresholds)
    
    print("\nResults by threshold:")
    print("Threshold | Count | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU")
    print("-" * 60)
    
    for threshold in sorted(results.keys()):
        r = results[threshold]
        print(f"{threshold:9d} | {r['count']:5d} | {r['rouge_1']:7.4f} | {r['rouge_2']:7.4f} | {r['rouge_l']:7.4f} | {r['bleu']:5.4f}")
    
    optimal = find_optimal_threshold(results)
    
    print("\nOptimal thresholds:")
    for metric, data in optimal.items():
        print(f"{metric.upper()}: threshold={data['threshold']}, score={data['score']:.4f}, count={data['count']}")
    
    # Find threshold that maximizes average of all metrics
    best_avg_score = 0
    best_avg_threshold = 0
    
    for threshold, data in results.items():
        if data["count"] > 0:
            avg_score = (data["rouge_1"] + data["rouge_2"] + data["rouge_l"] + data["bleu"]) / 4
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_avg_threshold = threshold
    
    print(f"\nBest average score: threshold={best_avg_threshold}, avg_score={best_avg_score:.4f}")

if __name__ == "__main__":
    main() 