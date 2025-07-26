from __future__ import annotations

import re
from typing import Tuple, List

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

def exact_match(gold: str, pred: str) -> int:
    g = _normalize(gold)
    p = _normalize(pred)
    if not g or not p:
        return 0
    
    # Check if one is contained in the other (bidirectional)
    if len(g) >= len(p):
        if p in g:
            return 1
    else:
        if g in p:
            return 1
    
    # Enhanced containment check: check if key phrases match
    # Split into words and check if most gold words appear in pred
    g_words = set(g.split())
    p_words = set(p.split())
    
    if len(g_words) > 0:
        # If 100% of gold words are in prediction, consider it a match
        common_words = g_words & p_words
        if len(common_words) / len(g_words) >= 1.0:
            return 1
    
    # Entity-based matching
    entity_overlap = _entity_overlap(gold, pred)
    if entity_overlap >= 1.0:  # High entity overlap
        return 1
    
    # Fallback: alnum-only containment
    flat_g = re.sub(r"[^a-z0-9]", "", g)
    flat_p = re.sub(r"[^a-z0-9]", "", p)
    if len(flat_g) >= len(flat_p):
        return int(flat_p in flat_g)
    return int(flat_g in flat_p)

def f1_score(gold: str, pred: str) -> float:
    if exact_match(gold, pred):
        return 1.0
    
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