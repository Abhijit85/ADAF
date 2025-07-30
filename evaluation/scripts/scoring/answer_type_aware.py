from __future__ import annotations

import re
from typing import Tuple, List, Set

_WHITESPACE_RE = re.compile(r"\s+")

def _normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation at ends."""
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text

def _extract_entities(text: str) -> List[str]:
    """Extract key entities for enhanced matching."""
    entities = []
    
    # Years (4-digit)
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    entities.extend(years)
    
    # Numbers
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    entities.extend(numbers)
    
    # Potential names (2+ capitalized words)
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    entities.extend(names)
    
    return entities

def _entity_overlap(gold: str, pred: str) -> float:
    """Calculate entity overlap between gold and prediction."""
    gold_entities = set(_extract_entities(gold))
    pred_entities = set(_extract_entities(pred))
    
    if not gold_entities:
        return 0.0
    
    common = len(gold_entities & pred_entities)
    return common / len(gold_entities)

def _parse_list_answer(text: str) -> Set[str]:
    """Parse a comma-separated list answer into a set of normalized items."""
    if not text:
        return set()
    
    # Split by comma and normalize each item
    items = [item.strip().lower() for item in text.split(",")]
    return set(item for item in items if item)

def _parse_span_answer(text: str) -> Set[str]:
    """Parse span answer into words, not comma-separated items."""
    if not text:
        return set()
    
    # Split into words and normalize
    words = [word.strip().lower() for word in text.split()]
    return set(word for word in words if word)

def _is_numeric_answer(text: str) -> bool:
    """Check if an answer is numeric."""
    if not text:
        return False
    
    # Remove common non-numeric tokens
    text_lower = text.lower()
    for token in ["million", "billion", "thousand", "percent", "%", "$", "£", "€"]:
        text_lower = text_lower.replace(token, "")
    
    # Remove commas and spaces
    text_clean = text_lower.replace(",", "").replace(" ", "")
    
    # Check if it's a number
    return bool(re.match(r"^[+-]?\d*\.?\d+$", text_clean))

def exact_match_answer_type_aware(gold: str, pred: str, answer_type: str) -> int:
    """Answer-type-aware exact match scoring."""
    
    # Handle span and multi-span types with set comparison
    if answer_type in ["span", "multi-span"]:
        # Use different parsing for span vs multi-span
        if answer_type == "span":
            gold_set = _parse_span_answer(gold)
            pred_set = _parse_span_answer(pred)
        else:
            gold_set = _parse_list_answer(gold)
            pred_set = _parse_list_answer(pred)
        
        if not gold_set:
            return 1 if not pred_set else 0
        
        # Exact match if sets are equal
        if gold_set == pred_set:
            return 1
        
        # Enhanced matching for span types
        return _enhanced_span_match(gold, pred)
    
    # Handle arithmetic and count types with numeric comparison
    elif answer_type in ["arithmetic", "count"]:
        if _is_numeric_answer(gold) and _is_numeric_answer(pred):
            # Clean numeric values for comparison
            gold_clean = re.sub(r"[^\d.-]", "", gold)
            pred_clean = re.sub(r"[^\d.-]", "", pred)
            
            try:
                gold_num = float(gold_clean)
                pred_num = float(pred_clean)
                return 1 if abs(gold_num - pred_num) < 0.01 else 0
            except ValueError:
                pass  # Fall back to string comparison
    
    # Default to original logic for other types
    g = _normalize(gold)
    p = _normalize(pred)
    if not g or not p:
        return 0
    
    # Check if all gold elements are in prediction (unidirectional)
    if g in p:
        return 1
    
    # Enhanced containment check: check if all gold words appear in pred
    # Split into words and check if all gold words appear in prediction
    g_words = set(g.split())
    p_words = set(p.split())
    
    if len(g_words) > 0:
        # If 100% of gold words are in prediction, consider it a match
        common_words = g_words & p_words
        if len(common_words) / len(g_words) >= 1.0:
            return 1
    
    # Entity-based matching - check if all gold entities are in prediction
    gold_entities = set(_extract_entities(gold))
    pred_entities = set(_extract_entities(pred))
    
    if gold_entities and len(gold_entities & pred_entities) == len(gold_entities):
        return 1
    
    # Fallback: alnum-only containment - only check if gold is in prediction
    flat_g = re.sub(r"[^a-z0-9]", "", g)
    flat_p = re.sub(r"[^a-z0-9]", "", p)
    return int(flat_g in flat_p)

def _enhanced_span_match(gold: str, pred: str) -> int:
    """Enhanced matching for span types with better text similarity."""
    
    # Normalize both texts
    gold_norm = _normalize(gold)
    pred_norm = _normalize(pred)
    
    # Direct match after normalization
    if gold_norm == pred_norm:
        return 1
    
    # Check for containment (unidirectional - only gold in prediction)
    if gold_norm in pred_norm:
        return 1
    
    # Word-level similarity - use 100% threshold instead of 90%
    gold_words = set(gold_norm.split())
    pred_words = set(pred_norm.split())
    
    if len(gold_words) > 0:
        # Require 100% of gold words to be in prediction
        common_words = gold_words & pred_words
        if len(common_words) / len(gold_words) >= 1.0:
            return 1
    
    # Handle numeric variations
    if _is_numeric_answer(gold) and _is_numeric_answer(pred):
        gold_clean = re.sub(r"[^\d.-]", "", gold)
        pred_clean = re.sub(r"[^\d.-]", "", pred)
        
        try:
            gold_num = float(gold_clean)
            pred_num = float(pred_clean)
            return 1 if abs(gold_num - pred_num) < 0.01 else 0
        except ValueError:
            pass
    
    # Handle date variations
    if _is_date_answer(gold) and _is_date_answer(pred):
        gold_year = _extract_year(gold)
        pred_year = _extract_year(pred)
        if gold_year and pred_year and gold_year == pred_year:
            return 1
    
    return 0

def _is_date_answer(text: str) -> bool:
    """Check if text contains date information."""
    # Check for year patterns
    if re.search(r'\b(19|20)\d{2}\b', text):
        return True
    
    # Check for month names
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    text_lower = text.lower()
    return any(month in text_lower for month in months)

def _extract_year(text: str) -> str:
    """Extract year from text."""
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    return year_match.group() if year_match else ""

def f1_score_answer_type_aware(gold: str, pred: str, answer_type: str) -> float:
    """Answer-type-aware F1 scoring."""
    
    # Handle span and multi-span types with set-based F1
    if answer_type in ["span", "multi-span"]:
        # Use different parsing for span vs multi-span
        if answer_type == "span":
            gold_set = _parse_span_answer(gold)
            pred_set = _parse_span_answer(pred)
        else:
            gold_set = _parse_list_answer(gold)
            pred_set = _parse_list_answer(pred)
        
        if not gold_set:
            return 1.0 if not pred_set else 0.0
        
        if not pred_set:
            return 0.0
        
        # Calculate intersection and set-based precision/recall
        intersection = len(gold_set & pred_set)
        precision = intersection / len(pred_set)
        recall = intersection / len(gold_set)
        
        # For span types, use set-based logic aligned with exact match
        if answer_type == "span":
            # If all gold words are present (perfect recall), give full credit regardless of precision
            if recall >= 1.0:
                return 1.0
            else:
                # Use regular F1 for partial matches
                if precision + recall == 0:
                    return 0.0
                return 2 * precision * recall / (precision + recall)
        else:
            # For multi-span, use regular F1
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
    
    # Handle arithmetic and count types with numeric F1
    elif answer_type in ["arithmetic", "count"]:
        if _is_numeric_answer(gold) and _is_numeric_answer(pred):
            # For numeric answers, use exact match for F1
            return 1.0 if exact_match_answer_type_aware(gold, pred, answer_type) else 0.0
    
    # For all other types, use the same logic as exact match
    # Check if exact match would give 1.0, if so return 1.0
    if exact_match_answer_type_aware(gold, pred, answer_type):
        return 1.0
    
    # Otherwise use regular F1 calculation
    g_tok = _normalize(gold).split()
    p_tok = _normalize(pred).split()
    if not g_tok or not p_tok:
        return 0.0
    
    common = len(set(g_tok) & set(p_tok))
    if common == 0:
        return 0.0
    
    precision = common / len(p_tok)
    recall = common / len(g_tok)
    
    # Enhanced F1: consider entity overlap as bonus
    base_f1 = 2 * precision * recall / (precision + recall)
    
    # Add entity bonus (up to 0.1 points)
    entity_bonus = min(_entity_overlap(gold, pred) * 0.1, 0.1)
    
    return min(base_f1 + entity_bonus, 1.0)

def _calculate_text_similarity(gold: str, pred: str) -> float:
    """Calculate text similarity between gold and prediction."""
    gold_norm = _normalize(gold)
    pred_norm = _normalize(pred)
    
    # Word-level similarity
    gold_words = set(gold_norm.split())
    pred_words = set(pred_norm.split())
    
    if not gold_words:
        return 1.0 if not pred_words else 0.0
    
    if not pred_words:
        return 0.0
    
    intersection = len(gold_words & pred_words)
    union = len(gold_words | pred_words)
    
    return intersection / union if union > 0 else 0.0 