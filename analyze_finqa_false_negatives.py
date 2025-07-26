#!/usr/bin/env python3

import json
from pathlib import Path
import re

def is_nearly_equal(a, b, rel_tol=0.02):
    try:
        a = float(a)
        b = float(b)
        if a == 0:
            return abs(b) <= rel_tol
        return abs(a - b) / abs(a) <= rel_tol
    except Exception:
        return False

def is_percentage_equivalent(gold, pred):
    try:
        if isinstance(gold, str) and '%' in gold:
            gold_num = float(gold.replace('%','').strip())
            pred_num = float(pred)
            if 0 <= pred_num <= 1:
                return abs(gold_num - pred_num*100) < 0.1
            if 0 <= pred_num <= 100:
                return abs(gold_num - pred_num) < 0.1
        if isinstance(pred, str) and '%' in pred:
            pred_num = float(pred.replace('%','').strip())
            gold_num = float(gold)
            if 0 <= gold_num <= 1:
                return abs(pred_num - gold_num*100) < 0.1
            if 0 <= gold_num <= 100:
                return abs(pred_num - gold_num) < 0.1
    except Exception:
        return False
    return False

def extract_number_and_unit(text):
    """Extract numeric value and unit from a string."""
    if not text:
        return None, None
    text = str(text)
    # Remove currency symbols and commas
    text = text.replace('$', '').replace(',', '')
    
    # Find number
    num_match = re.search(r'-?\d+(?:\.\d+)?', text)
    num = float(num_match.group()) if num_match else None
    
    # Find unit
    unit_match = re.search(r'(million|billion|thousand|%)', text, re.IGNORECASE)
    unit = unit_match.group(1).lower() if unit_match else None
    
    return num, unit

def normalize_to_base_number(num, unit):
    """Convert number and unit to base number."""
    if num is None:
        return 0.0
    if unit == 'million':
        return num * 1_000_000
    if unit == 'billion':
        return num * 1_000_000_000
    if unit == 'thousand':
        return num * 1_000
    if unit == '%':
        return num / 100
    return num

def analyze_case(rec):
    """Analyze a single case and categorize it."""
    gold_answer = rec.get("gold_answer")
    gold_exe_ans = rec.get("gold_exe_ans")
    pred_raw = rec.get("pred_raw") or rec.get("pred_original")
    qid = rec.get("qid")
    
    # Check if prediction is empty/null
    if not pred_raw or str(pred_raw).strip() in ['', 'None', 'null', 'nan']:
        return "true_model_error", "Empty prediction"
    
    gold_str = str(gold_answer)
    pred_str = str(pred_raw)
    
    # Case 1: Both have units (e.g., "$2.33 million" vs "$2.87 million")
    gold_num, gold_unit = extract_number_and_unit(gold_str)
    pred_num, pred_unit = extract_number_and_unit(pred_str)
    
    if gold_unit and pred_unit and gold_unit == pred_unit:
        if is_nearly_equal(gold_num, pred_num):
            return "false_negative", f"Unit match: {gold_num} vs {pred_num} {gold_unit}"
        else:
            return "true_model_error", f"Wrong value: {gold_num} vs {pred_num} {gold_unit}"
    
    # Case 2: Gold has unit, pred doesn't (e.g., "$2.33 million" vs "2.87")
    if gold_unit and not pred_unit:
        gold_base = normalize_to_base_number(gold_num, gold_unit)
        try:
            pred_base = float(pred_str)
            if is_nearly_equal(gold_base, pred_base):
                return "false_negative", f"Unit scaling: {gold_base} vs {pred_base}"
            else:
                return "true_model_error", f"Wrong scaled value: {gold_base} vs {pred_base}"
        except ValueError:
            return "true_model_error", "Prediction not numeric"
    
    # Case 3: Pred has unit, gold doesn't
    if not gold_unit and pred_unit:
        pred_base = normalize_to_base_number(pred_num, pred_unit)
        try:
            gold_base = float(gold_str)
            if is_nearly_equal(gold_base, pred_base):
                return "false_negative", f"Unit scaling: {gold_base} vs {pred_base}"
            else:
                return "true_model_error", f"Wrong scaled value: {gold_base} vs {pred_base}"
        except ValueError:
            return "true_model_error", "Gold not numeric"
    
    # Case 4: Percentage vs decimal
    if is_percentage_equivalent(gold_str, pred_str):
        return "false_negative", f"Percentage/decimal mismatch: {gold_str} vs {pred_str}"
    
    # Case 5: Sign flip
    try:
        gold_num = float(gold_str)
        pred_num = float(pred_str)
        if gold_num == -pred_num:
            return "false_negative", f"Sign flip: {gold_num} vs {pred_num}"
    except ValueError:
        pass
    
    # Case 6: Direct numeric comparison
    try:
        gold_num = float(gold_str)
        pred_num = float(pred_str)
        if is_nearly_equal(gold_num, pred_num):
            return "false_negative", f"Close numeric: {gold_num} vs {pred_num}"
    except ValueError:
        pass
    
    # Case 7: String contains gold or gold contains pred
    if gold_str in pred_str or pred_str in gold_str:
        return "false_negative", f"Substring match: {gold_str} vs {pred_str}"
    
    return "true_model_error", f"Unmatched: {gold_str} vs {pred_str}"

def main():
    scored_file = Path("evaluation/finqa/deepseek-r1-671b_lambda_ee_fs_20250725_033338/scored_20250726_030610.json")
    with open(scored_file, 'r') as f:
        data = json.load(f)
    
    failed = [rec for rec in data if rec.get("exact_match", 1) == 0 and rec.get("f1", 1.0) == 0.0]
    print(f"Total failed cases: {len(failed)}\n")
    
    categories = {
        "false_negative": [],
        "true_model_error": []
    }
    
    for rec in failed:
        category, reason = analyze_case(rec)
        categories[category].append((rec, reason))
    
    print(f"False negatives (evaluation issues): {len(categories['false_negative'])}")
    print(f"True model errors: {len(categories['true_model_error'])}\n")
    
    # Show examples of false negatives
    print("=== FALSE NEGATIVES (Need Code Fixes) ===")
    for i, (rec, reason) in enumerate(categories['false_negative'][:10]):
        print(f"{i+1}. {rec.get('qid')}")
        print(f"   Gold: {rec.get('gold_answer')} | {rec.get('gold_exe_ans')}")
        print(f"   Pred: {rec.get('pred_raw')}")
        print(f"   Reason: {reason}")
        print()
    
    # Show examples of true model errors
    print("=== TRUE MODEL ERRORS ===")
    for i, (rec, reason) in enumerate(categories['true_model_error'][:10]):
        print(f"{i+1}. {rec.get('qid')}")
        print(f"   Gold: {rec.get('gold_answer')} | {rec.get('gold_exe_ans')}")
        print(f"   Pred: {rec.get('pred_raw')}")
        print(f"   Reason: {reason}")
        print()
    
    # Detailed analysis of specific patterns
    print("=== DETAILED PATTERN ANALYSIS ===")
    
    # Unit scaling patterns
    unit_scaling_cases = [(r, reason) for r, reason in categories['false_negative'] if 'Unit scaling' in reason]
    print(f"Unit scaling issues: {len(unit_scaling_cases)}")
    for rec, reason in unit_scaling_cases[:3]:
        print(f"  {rec.get('qid')}: {reason}")
    
    # Percentage/decimal patterns
    percentage_cases = [(r, reason) for r, reason in categories['false_negative'] if 'Percentage/decimal' in reason]
    print(f"Percentage/decimal issues: {len(percentage_cases)}")
    for rec, reason in percentage_cases[:3]:
        print(f"  {rec.get('qid')}: {reason}")
    
    # Sign flip patterns
    sign_flip_cases = [(r, reason) for r, reason in categories['false_negative'] if 'Sign flip' in reason]
    print(f"Sign flip issues: {len(sign_flip_cases)}")
    for rec, reason in sign_flip_cases[:3]:
        print(f"  {rec.get('qid')}: {reason}")

if __name__ == "__main__":
    main() 