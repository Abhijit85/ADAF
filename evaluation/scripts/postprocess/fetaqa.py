from __future__ import annotations

import re
from typing import List, Dict, Any

# Less aggressive patterns - preserve more semantic content
_PUNCT_TO_SPACE = re.compile(r'[^\w\s]')  # only remove pure punctuation, keep alphanumeric
_MULTI_SPACE = re.compile(r'\s+')

# Content cleaning patterns
_TABLE_METADATA = re.compile(r'the following table lists?|the table shows?|according to the table', re.I)
_ASSUMPTION_PHRASES = re.compile(r'assuming.*?table|based on.*?data|according to.*?data', re.I)
_FILLER_WORDS = re.compile(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', re.I)

def _clean_content(text: str) -> str:
    """Remove table metadata and assumption phrases from gold answers."""
    if text is None:
        return ""
    
    # Remove table metadata
    text = _TABLE_METADATA.sub('', text)
    
    # Remove assumption phrases
    text = _ASSUMPTION_PHRASES.sub('', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def _extract_key_entities(text: str) -> List[str]:
    """Extract key entities for matching."""
    if text is None:
        return []
    
    # Extract years (4-digit numbers)
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    
    # Extract numbers
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    
    # Extract potential names (capitalized words)
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    return years + numbers + names

def _norm(s: str) -> str:
    """Improved normalisation that preserves more semantic content."""
    if not s or s is None:
        return ""
    
    # lowercase and trim
    s = s.lower().strip()
    
    # Replace punctuation with spaces (not remove completely)
    s = _PUNCT_TO_SPACE.sub(' ', s)
    
    # Collapse multiple spaces
    s = _MULTI_SPACE.sub(' ', s)
    
    return s.strip()

class FetaqaPostProcessor:
    """Enhanced post-processing for FETA-QA with content extraction and cleaning."""

    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in records:
            gold_raw = rec.get("gold_answer", "")
            pred_raw = rec.get("pred_answer", "")
            
            # Clean gold answer (remove table metadata, assumptions)
            gold_cleaned = _clean_content(gold_raw)
            
            # Normalize both
            pp_gold = _norm(gold_cleaned)
            pp_pred = _norm(pred_raw)
            
            # Extract key entities for additional matching
            gold_entities = _extract_key_entities(gold_raw)
            pred_entities = _extract_key_entities(pred_raw)
            
            rec["pp_gold_answer"] = pp_gold
            rec["pp_pred_answer"] = pp_pred
            rec["gold_entities"] = gold_entities
            rec["pred_entities"] = pred_entities
            
            out.append(rec)
        return out 