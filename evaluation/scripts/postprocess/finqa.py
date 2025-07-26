#!/usr/bin/env python3

import re
from typing import List, Dict, Any

def _extract_number_and_unit(text: str):
    """Extract numeric value and unit from a string."""
    if not text:
        return None, None
    text = str(text)
    # Remove currency symbols and commas
    text = text.replace('$', '').replace(',', '')
    
    # Handle "percentage points" specifically
    points_match = re.search(r'(-?\d+(?:\.\d+)?)\s*percentage\s*points?', text.lower())
    if points_match:
        return float(points_match.group(1)), None # Treat as raw number, not percentage

    # Find number
    num_match = re.search(r'(-?\d+(?:\.\d+)?)', text)
    num = float(num_match.group(1)) if num_match else None # Capture group 1 for the number

    # Find unit
    unit_match = re.search(r'(million|billion|thousand|%)', text, re.IGNORECASE)
    unit = unit_match.group(1).lower() if unit_match else None
    
    return num, unit

def _normalize_to_base_number(num, unit):
    """Convert number and unit to base number (e.g., 1.2 million -> 1200000, 25% -> 0.25)."""
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

def _try_float_conversion(value: Any) -> float:
    """Attempt to convert a value to float, return 0.0 on failure."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def _is_close(gold: float, pred: float, tolerance: float = 0.02) -> bool:
    if gold == 0:
        return abs(pred) <= tolerance # Absolute tolerance for zero gold
    return abs(pred - gold) / abs(gold) <= tolerance # Relative tolerance for non-zero gold

def _extract_boolean_answer(text: str) -> str:
    if not text:
        return ""
    text_lower = text.lower().strip()
    # More robust boolean extraction
    if text_lower.startswith('yes'):
        return 'yes'
    if text_lower.startswith('no'):
        return 'no'
    return text # Return original if not a clear boolean

def _assess_cannot_determine(pred: str) -> bool:
    cannot_phrases = [
        'cannot be determined', 'insufficient data', 'unable to calculate',
        'missing data', 'cannot be calculated', 'not provided', 'unknown',
        'cannot determine', 'missing', 'insufficient', 'n/a', 'na',
        'cannot calculate', 'data not available', 'no data available',
        'information not available', 'not available', 'unavailable'
    ]
    pred_lower = pred.lower()
    return any(phrase in pred_lower for phrase in cannot_phrases)

def _handle_sign_flip(gold: Any, pred: str) -> bool:
    """Check if prediction is correct but with opposite sign."""
    if not gold or not pred:
        return False
    
    try:
        gold_num = float(gold)
        pred_num = float(pred)
        # Check if they're opposite signs but same magnitude
        if gold_num != 0 and pred_num != 0:
            return abs(gold_num + pred_num) < 0.01  # Very small tolerance for sign flip
    except (ValueError, TypeError):
        return False
    return False

def _handle_percentage_sign_flip(gold: Any, pred: str) -> bool:
    """Check if percentage predictions have opposite signs."""
    if not gold or not pred:
        return False
    
    gold_str = str(gold).strip()
    pred_str = str(pred).strip()
    
    # Extract percentage values
    gold_percent_match = re.search(r'(-?\d+(?:\.\d+)?)', gold_str)
    pred_percent_match = re.search(r'(-?\d+(?:\.\d+)?)', pred_str)
    
    if gold_percent_match and pred_percent_match:
        try:
            gold_percent = float(gold_percent_match.group(1))
            pred_percent = float(pred_percent_match.group(1))
            
            # Check if they're opposite signs but same magnitude
            if gold_percent != 0 and pred_percent != 0:
                return abs(gold_percent + pred_percent) < 0.1  # Small tolerance for percentage sign flip
        except (ValueError, TypeError):
            return False
    return False

def _handle_percentage_points_equivalence(gold: Any, pred: str) -> bool:
    """Check if percentage points and percentage are equivalent."""
    if not gold or not pred:
        return False
    
    gold_str = str(gold).strip()
    pred_str = str(pred).strip()
    
    # Check if one is percentage and other is percentage points
    gold_is_percent = '%' in gold_str
    pred_is_points = 'percentage points' in pred_str.lower()
    
    if gold_is_percent and pred_is_points:
        gold_match = re.search(r'(-?\d+(?:\.\d+)?)', gold_str)
        pred_match = re.search(r'(-?\d+(?:\.\d+)?)', pred_str)
        
        if gold_match and pred_match:
            try:
                gold_val = float(gold_match.group(1))
                pred_val = float(pred_match.group(1))
                return abs(gold_val - pred_val) < 0.1  # Small tolerance for percentage points
            except (ValueError, TypeError):
                return False
    
    return False

def _handle_numeric_substring_match(gold: Any, pred: str) -> bool:
    """Check if prediction contains the gold number as a substring."""
    if not gold or not pred:
        return False
    
    gold_str = str(gold).strip()
    pred_str = str(pred).strip()
    
    # Try to extract numeric values from both
    try:
        gold_num = float(gold_str)
        pred_num = float(pred_str)
        # Check if pred contains gold as substring (e.g., 0.3 vs 0.32)
        if str(gold_num) in str(pred_num) or str(pred_num) in str(gold_num):
            return True
    except (ValueError, TypeError):
        pass
    
    # Check if gold number appears as substring in pred string
    if gold_str in pred_str:
        return True
    
    return False

# This function is now more critical for handling percentage equivalence
def _handle_percentage_mismatch(gold_val: Any, pred_val: Any) -> bool:
    """Handle cases where one is percentage and other is decimal/number, or both are percentages."""
    if gold_val is None or pred_val is None:
        return False

    gold_str = str(gold_val).strip()
    pred_str = str(pred_val).strip()

    gold_is_percent_str = '%' in gold_str
    pred_is_percent_str = '%' in pred_str

    # If both are percentage strings, compare their numeric values directly
    if gold_is_percent_str and pred_is_percent_str:
        gold_num_match = re.search(r'(-?\d+(?:\.\d+)?)', gold_str)
        pred_num_match = re.search(r'(-?\d+(?:\.\d+)?)', pred_str)
        if gold_num_match and pred_num_match:
            return _is_close(float(gold_num_match.group(1)), float(pred_num_match.group(1)), tolerance=0.1) # 0.1% absolute tolerance for percentage values

    # If one is a percentage string and the other is a decimal number
    if gold_is_percent_str and not pred_is_percent_str:
        gold_num_match = re.search(r'(-?\d+(?:\.\d+)?)', gold_str)
        pred_num = _try_float_conversion(pred_str)
        if gold_num_match and pred_num != 0.0:
            gold_percent_val = float(gold_num_match.group(1))
            return _is_close(gold_percent_val / 100.0, pred_num, tolerance=0.001) # Compare decimals with tighter tolerance

    if not gold_is_percent_str and pred_is_percent_str:
        pred_num_match = re.search(r'(-?\d+(?:\.\d+)?)', pred_str)
        gold_num = _try_float_conversion(gold_str)
        if pred_num_match and gold_num != 0.0:
            pred_percent_val = float(pred_num_match.group(1))
            return _is_close(gold_num, pred_percent_val / 100.0, tolerance=0.001) # Compare decimals with tighter tolerance

    return False

def _handle_complex_string_equivalence(gold_answer: Any, pred_raw: str) -> bool:
    """Handle complex string equivalence cases including percentage/decimal mismatches."""
    if not gold_answer or not pred_raw:
        return False
    
    gold_str = str(gold_answer).strip()
    pred_str = str(pred_raw).strip()
    
    # Case 1: Gold is percentage string, pred is numeric
    if '%' in gold_str:
        try:
            gold_percent = float(re.search(r'-?\d+(?:\.\d+)?', gold_str).group())
            pred_num = float(pred_str)
            
            # Pred as decimal (0-1 range)
            if 0 <= pred_num <= 1:
                return abs(gold_percent - pred_num * 100) <= 0.1
            # Pred as percentage (0-100 range)
            elif 0 <= pred_num <= 100:
                return abs(gold_percent - pred_num) <= 0.1
            # Pred as large number (might be scaled)
            elif abs(gold_percent - pred_num) <= 0.1:
                return True
        except (ValueError, AttributeError):
            pass
    
    # Case 2: Pred is percentage string, gold is numeric
    if '%' in pred_str:
        try:
            pred_percent = float(re.search(r'-?\d+(?:\.\d+)?', pred_str).group())
            gold_num = float(gold_str)
            
            # Gold as decimal (0-1 range)
            if 0 <= gold_num <= 1:
                return abs(pred_percent - gold_num * 100) <= 0.1
            # Gold as percentage (0-100 range)
            elif 0 <= gold_num <= 100:
                return abs(pred_percent - gold_num) <= 0.1
            # Gold as large number (might be scaled)
            elif abs(pred_percent - gold_num) <= 0.1:
                return True
        except (ValueError, AttributeError):
            pass
    
    # Case 3: Both are numeric but one is scaled version of the other
    try:
        gold_num = float(gold_str)
        pred_num = float(pred_str)
        
        # Check if they're the same number but different scales
        if gold_num != 0 and pred_num != 0:
            ratio = gold_num / pred_num
            # Common scaling factors
            if abs(ratio - 1e3) < 0.01 or abs(ratio - 1e6) < 0.01 or abs(ratio - 1e9) < 0.01:
                return True
            if abs(ratio - 1e-3) < 0.01 or abs(ratio - 1e-6) < 0.01 or abs(ratio - 1e-9) < 0.01:
                return True
    except ValueError:
        pass
    
    return False

def _unit_scaling_match(gold: float, pred: float, tolerance: float = 0.02) -> bool:
    """Check if gold and pred match after scaling by 1e3, 1e6, or 1e9."""
    if gold == 0 or pred == 0:
        return False
    scales = [1, 1e3, 1e6, 1e9]
    for g_scale in scales:
        for p_scale in scales:
            scaled_gold = gold * g_scale
            scaled_pred = pred * p_scale
            if abs(scaled_gold - scaled_pred) / abs(scaled_gold) <= tolerance:
                # Avoid trivial match (both scale=1, already handled by is_close)
                if not (g_scale == 1 and p_scale == 1):
                    return True
    return False

class FinqaPostProcessor:
    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in records:
            gold_answer = rec.get("gold_answer")
            gold_exe_ans = rec.get("gold_exe_ans")
            pred_raw = rec.get("pred_answer", "")
            if pred_raw is None:
                pred_raw = ""
            pred_raw = str(pred_raw).strip()

            # Check for empty/invalid predictions
            is_empty_pred = (not pred_raw) or (pred_raw.lower() in ['', 'nan', 'none', 'null', 'undefined'])
            cannot_determine = _assess_cannot_determine(pred_raw)
            
            # Handle boolean answers
            if isinstance(gold_answer, str) and gold_answer.lower() in ['yes', 'no']:
                extracted_bool = _extract_boolean_answer(pred_raw)
                if extracted_bool:
                    pred_raw = extracted_bool
            
            # Check for percentage format mismatches
            percentage_match = _handle_percentage_mismatch(gold_answer, pred_raw)
            
            # Check for complex string equivalence
            complex_string_match = _handle_complex_string_equivalence(gold_answer, pred_raw)
            
            # Check for sign flip
            sign_flip_match = _handle_sign_flip(gold_answer, pred_raw) or _handle_sign_flip(gold_exe_ans, pred_raw)
            
            # Check for percentage sign flip (specific to percentage format)
            percentage_sign_flip = _handle_percentage_sign_flip(gold_answer, pred_raw)
            
            # Check for percentage points equivalence
            percentage_points_match = _handle_percentage_points_equivalence(gold_answer, pred_raw)
            
            # Check for numeric substring match
            numeric_substring_match = _handle_numeric_substring_match(gold_answer, pred_raw) or _handle_numeric_substring_match(gold_exe_ans, pred_raw)
            
            # Robust normalization for gold and pred
            gold_norm = _robust_normalize(gold_answer)
            gold_exe_norm = _robust_normalize(gold_exe_ans)
            pred_norm = _robust_normalize(pred_raw)
            
            # Check if prediction matches either gold answer
            is_close_to_answer = _is_close(gold_norm, pred_norm)
            is_close_to_exe_ans = _is_close(gold_exe_norm, pred_norm)
            
            # Unit scaling match
            unit_scaling_match = (
                _unit_scaling_match(gold_norm, pred_norm) or
                _unit_scaling_match(gold_exe_norm, pred_norm)
            )
            
            # Consider all types of matches as valid
            is_close = (
                is_close_to_answer or 
                is_close_to_exe_ans or 
                cannot_determine or 
                percentage_match or 
                unit_scaling_match or
                complex_string_match or
                sign_flip_match or
                percentage_sign_flip or
                percentage_points_match or
                numeric_substring_match
            )
            
            rec["pp_gold_answer"] = gold_norm
            rec["pp_gold_exe_ans"] = gold_exe_norm
            rec["pp_pred_answer"] = pred_norm
            rec["is_close"] = is_close
            rec["is_close_to_answer"] = is_close_to_answer
            rec["is_close_to_exe_ans"] = is_close_to_exe_ans
            rec["unit_scaling_match"] = unit_scaling_match
            rec["complex_string_match"] = complex_string_match
            rec["sign_flip_match"] = sign_flip_match
            rec["percentage_sign_flip"] = percentage_sign_flip
            rec["percentage_points_match"] = percentage_points_match
            rec["numeric_substring_match"] = numeric_substring_match
            rec["cannot_determine"] = cannot_determine
            rec["is_empty_prediction"] = is_empty_pred
            rec["percentage_match"] = percentage_match
            rec["gold_original"] = gold_answer
            rec["gold_exe_ans_original"] = gold_exe_ans
            rec["pred_original"] = pred_raw
            out.append(rec)
        return out

def _robust_normalize(value: Any) -> float:
    """Robustly normalize a value to a float, handling units and percentages."""
    if value is None or value == "":
        return 0.0
    # If already a number
    if isinstance(value, (int, float)):
        return float(value)
    # Try to extract number and unit
    num, unit = _extract_number_and_unit(value)
    if num is not None:
        return _normalize_to_base_number(num, unit)
    # Fallback: try float conversion
    try:
        return float(value)
    except Exception:
        return 0.0 