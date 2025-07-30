#!/usr/bin/env python3
"""Test script to validate FETAQA-specific normalization improvements."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scoring'))

from scoring.nlg_metrics import (
    normalize_for_fetaqa_specific,
    calculate_nlg_metrics_fetaqa,
    calculate_nlg_metrics
)

def test_normalization():
    """Test the FETAQA-specific normalization on example cases."""
    
    test_cases = [
        {
            "name": "Tony Awards Example",
            "gold": "Jennifer Ehle won an Award for Actress in a Play for The Real Thing in 2000 and an Award for Featured Actress in a Play for The Coast of Utopia in 2007.",
            "pred": "Jennifer Ehle won two Tony Awards: the Tony Award for Actress in a Play in 2000 for her role in *The Real Thing*, and the Tony Award for Featured Actress in a Play in 2007 for her performance in *The Coast of Utopia*."
        },
        {
            "name": "CPI Example",
            "gold": "Smarajit Bandopadhyay and Promatha Ranjan Thakur, both of Congress, won the Haringhata joint seat in 1957.",
            "pred": "Smarajit Bandopadhyay of the Indian National Congress won the Chapra seat in 1951 and the Haringhata seat in 1957 during the 1950s."
        },
        {
            "name": "Boeing Example",
            "gold": "The 777-200LR and 777-300ER expansion were the first PMDG products which were released on February 7, 2015.",
            "pred": "No expansions were released on February 15. In February 2015, the Boeing 777-200LR/F and Boeing 777-300ER Expansion were released on February 7, 2015 for Prepar3D V2/V3/v4."
        },
        {
            "name": "Tinker Bell Example",
            "gold": "In 2009, Chenoweth had a voice role as Rosetta in Tinker Bell and the Lost Treasure and reprised the role in Tinker Bell and the Great Fairy Rescue 2010.",
            "pred": "Kristin Chenoweth appeared in three Tinker Bell films as the voice of Rosetta: *Tinker Bell* (2008), *Tinker Bell and the Lost Treasure* (2009), and *Tinker Bell and the Great Fairy Rescue* (2010)."
        }
    ]
    
    print("Testing FETAQA-specific normalization improvements...")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['name']}")
        print("-" * 40)
        
        gold_norm = normalize_for_fetaqa_specific(case['gold'])
        pred_norm = normalize_for_fetaqa_specific(case['pred'])
        
        print(f"Original Gold: {case['gold']}")
        print(f"Normalized Gold: {gold_norm}")
        print(f"Original Pred: {case['pred']}")
        print(f"Normalized Pred: {pred_norm}")
        
        # Calculate metrics with both old and new normalization
        old_metrics = calculate_nlg_metrics(case['gold'], case['pred'], min_length=30)
        new_metrics = calculate_nlg_metrics_fetaqa(case['gold'], case['pred'], min_length=30)
        
        print(f"\nOld Metrics: ROUGE-1={old_metrics['rouge_1']:.4f}, ROUGE-2={old_metrics['rouge_2']:.4f}, ROUGE-L={old_metrics['rouge_l']:.4f}, BLEU={old_metrics['bleu']:.4f}")
        print(f"New Metrics: ROUGE-1={new_metrics['rouge_1']:.4f}, ROUGE-2={new_metrics['rouge_2']:.4f}, ROUGE-L={new_metrics['rouge_l']:.4f}, BLEU={new_metrics['bleu']:.4f}")
        
        # Calculate improvements
        rouge_1_improvement = new_metrics['rouge_1'] - old_metrics['rouge_1']
        rouge_2_improvement = new_metrics['rouge_2'] - old_metrics['rouge_2']
        rouge_l_improvement = new_metrics['rouge_l'] - old_metrics['rouge_l']
        bleu_improvement = new_metrics['bleu'] - old_metrics['bleu']
        
        print(f"Improvements: ROUGE-1={rouge_1_improvement:+.4f}, ROUGE-2={rouge_2_improvement:+.4f}, ROUGE-L={rouge_l_improvement:+.4f}, BLEU={bleu_improvement:+.4f}")

if __name__ == "__main__":
    test_normalization() 