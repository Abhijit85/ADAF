from __future__ import annotations

import re
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fetaqa import FetaqaPostProcessor

# Import the same patterns from FETA-QA post-processor
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

class FetaqaPerturbedPostProcessor(FetaqaPostProcessor):
    """Enhanced post-processor for perturbed FETA-QA data."""
    
    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process records with perturbation-aware logic."""
        out = []
        
        for rec in records:
            # Apply standard FETA-QA post-processing
            processed_rec = super().process_records([rec])[0]
            
            # Preserve perturbation metadata
            perturbation_metadata = {
                "perturbation_class": rec.get("perturbation_class", ""),
                "perturbation_type": rec.get("perturbation_type", ""),
                "modality": rec.get("modality", ""),
                "original_id": rec.get("original_id", ""),
                "perturbation_notes": rec.get("perturbation_notes", "")
            }
            
            # Add perturbation metadata to processed record
            for key, value in perturbation_metadata.items():
                processed_rec[key] = value
            
            # Apply perturbation-specific processing
            perturbation_class = rec.get("perturbation_class", "")
            perturbation_type = rec.get("perturbation_type", "")
            
            if perturbation_class == "invariant":
                processed_rec = self._process_invariant_perturbation(processed_rec)
            elif perturbation_class == "counterfactual":
                processed_rec = self._process_counterfactual_perturbation(processed_rec)
            elif perturbation_class == "unanswerable":
                processed_rec = self._process_unanswerable_perturbation(processed_rec)
            
            out.append(processed_rec)
        
        return out
    
    def _process_invariant_perturbation(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Special processing for invariant perturbations."""
        # For invariant perturbations, we want to ensure the same scoring applies
        # No special processing needed - standard FETA-QA processing should work
        return rec
    
    def _process_counterfactual_perturbation(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Special processing for counterfactual perturbations."""
        # For counterfactual perturbations, we need to handle the fact that
        # the gold answer might have changed
        gold_answer = rec.get("gold_answer", "")
        pred_answer = rec.get("pred_answer", "")
        
        # If the perturbation changed numeric values, ensure proper comparison
        if "change_numeric_cell" in rec.get("perturbation_type", ""):
            # Ensure numeric values are properly normalized for comparison
            rec["pp_gold_answer"] = _norm(gold_answer)
            rec["pp_pred_answer"] = _norm(pred_answer)
        
        return rec
    
    def _process_unanswerable_perturbation(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Special processing for unanswerable perturbations."""
        # For unanswerable perturbations, we need to handle "cannot answer" responses
        gold_answer = rec.get("gold_answer", "")
        pred_answer = rec.get("pred_answer", "")
        
        # Check if gold answer indicates unanswerable
        if "cannot answer" in gold_answer.lower() or "missing" in gold_answer.lower():
            # If prediction also indicates unanswerable, give partial credit
            if any(phrase in pred_answer.lower() for phrase in ["cannot", "missing", "insufficient", "unavailable"]):
                # This will be handled by the scoring logic
                pass
        
        return rec 