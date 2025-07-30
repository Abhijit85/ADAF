from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List

try:
    from evaluation.scripts.base_dataset import BaseDataset  # type: ignore
except ModuleNotFoundError:
    from base_dataset import BaseDataset  # type: ignore

class FetaqaPerturbedDataset(BaseDataset):
    """Enhanced FETA-QA dataset that supports perturbed examples."""
    
    def __init__(self, perturbation_class: str = None, perturbation_type: str = None):
        # Try to get from environment variables if not provided directly
        import os
        self.perturbation_class = perturbation_class or os.environ.get("PERTURBATION_CLASS")
        self.perturbation_type = perturbation_type or os.environ.get("PERTURBATION_TYPE")
        self.GOLD_FILE = self._get_gold_file_path()
    
    def _get_gold_file_path(self) -> Path:
        """Get the path to the gold file based on perturbation settings."""
        base_path = Path(__file__).resolve().parents[3] / "examples" / "fetaqa_perturbed"
        
        if self.perturbation_class and self.perturbation_type:
            # Specific perturbation type
            return base_path / self.perturbation_class / self.perturbation_type
        else:
            # All perturbed data
            return base_path
    
    def load_gold(self) -> Dict[str, Dict[str, Any]]:
        """Load gold answers for perturbed dataset."""
        gold = {}
        gold_path = self.GOLD_FILE
        
        if not gold_path.exists():
            print(f"Warning: Gold file path does not exist: {gold_path}")
            return gold
        
        # Load perturbed examples recursively
        if gold_path.is_dir():
            # Recursively find all JSON files in perturbation directories
            json_files = list(gold_path.rglob("*.json"))
            print(f"Found {len(json_files)} perturbed example files")
            
            for json_file in sorted(json_files):
                try:
                    with json_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Extract perturbation info from path
                    # When we're in a specific perturbation directory, the path structure is different
                    if self.perturbation_class and self.perturbation_type:
                        # We're already in the specific perturbation directory, so use the stored values
                        file_perturbation_class = self.perturbation_class
                        file_perturbation_type = self.perturbation_type
                    else:
                        # We're in the root directory, so extract from path
                        path_parts = json_file.relative_to(gold_path).parts
                        file_perturbation_class = path_parts[0] if len(path_parts) > 0 else ""
                        file_perturbation_type = path_parts[1] if len(path_parts) > 1 else ""
                    
                    # Filter by perturbation type if specified
                    if self.perturbation_class and self.perturbation_type:
                        if (file_perturbation_class != self.perturbation_class or 
                            file_perturbation_type != self.perturbation_type):
                            continue
                    
                    # Extract feta_id and question from original_data section
                    original_data = data.get("original_data", {})
                    base_feta_id = original_data.get("feta_id", f"unknown_{json_file.stem}")
                    
                    # Create unique ID that includes perturbation info to avoid overwriting
                    qid = f"{file_perturbation_class}_{file_perturbation_type}_{base_feta_id}"
                    
                    # Get the new gold answer - use new_gold field which contains the perturbed answer
                    new_gold = data.get("new_gold", "")
                    
                    # Get the question - it's in the original_data section
                    question = original_data.get("question", "")
                    
                    # Get original_id for matching
                    original_id = data.get("original_id", "")
                    
                    gold[qid] = {
                        "answer": new_gold,  # Use new_gold which contains the perturbed answer
                        "perturbation_class": file_perturbation_class,
                        "perturbation_type": file_perturbation_type,
                        "modality": data.get("modality", ""),
                        "original_id": original_id,
                        "perturbation_notes": data.get("perturbation_notes", ""),
                        "question": question,
                        "table_array": original_data.get("table_array", []),
                        "highlighted_cell_ids": original_data.get("highlighted_cell_ids", [])
                    }
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")
                    continue
        else:
            # Single file
            try:
                with gold_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    original_data = data.get("original_data", {})
                    qid = original_data.get("feta_id", "unknown")
                    
                    gold[qid] = {
                        "answer": data.get("new_gold", ""),  # Use new_gold field
                        "perturbation_class": data.get("perturbation_class", ""),
                        "perturbation_type": data.get("perturbation_type", ""),
                        "modality": data.get("modality", ""),
                        "original_id": data.get("original_id", ""),
                        "perturbation_notes": data.get("perturbation_notes", ""),
                        "question": original_data.get("question", ""),
                        "table_array": original_data.get("table_array", []),
                        "highlighted_cell_ids": original_data.get("highlighted_cell_ids", [])
                    }
            except Exception as e:
                print(f"Warning: Failed to load {gold_path}: {e}")
        
        print(f"Loaded {len(gold)} perturbed examples")
        return gold
    
    def _parse_single_file(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """Parse a single output file for perturbed data."""
        text = path.read_text(encoding="utf-8", errors="ignore")
        if self._FS_MARKER not in text:
            raise ValueError("marker not found")
        
        summary_part = text.split(self._FS_MARKER, 1)[1]
        # collect lines until next big section or Answer Echoes
        lines = []
        for line in summary_part.splitlines()[1:]:
            if line.startswith("=== ") or line.startswith("Answer Echoes"):
                break
            lines.append(line)

        cleaned = "\n".join(lines).strip()
        cleaned = re.sub(r"```json|```", "", cleaned, flags=re.IGNORECASE).strip()

        # balanced brace extraction
        start = cleaned.find("{")
        if start == -1:
            raise ValueError("JSON opening brace not found")
        depth = 0
        end = -1
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end == -1:
            raise ValueError("Unbalanced braces")
        json_str = cleaned[start:end+1]
        json_str = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", json_str)
        obj = json.loads(json_str, strict=False)

        q = str(obj.get("Question", "")).strip()
        if not q:
            raise ValueError("Question empty")
        return {
            q: {
                "answer": str(obj.get("Factual Answer", "")).strip(),
                "source": str(obj.get("Source", "")).strip(),
                "confidence": str(obj.get("Confidence", "")).strip(),
            }
        }
    
    def parse_run_dir(self, run_dir: str | Path) -> Dict[str, Dict[str, Any]]:
        """Parse predictions from run directory for perturbed data."""
        run_path = Path(run_dir)
        if not run_path.is_dir():
            raise FileNotFoundError(run_dir)

        preds: Dict[str, Dict[str, Any]] = {}
        successful_parses = 0
        failed_parses = 0
        
        for txt in sorted(run_path.glob("*_out.txt")):
            try:
                # Extract original_id and perturbation info from filename
                # Format: perturbation_class_perturbation_type_original_id_out.txt
                filename_parts = txt.stem.split('_')
                if len(filename_parts) >= 4:
                    perturbation_class = filename_parts[0]
                    # Handle multi-word perturbation types like "change_numeric_cell"
                    if len(filename_parts) >= 5:
                        perturbation_type = "_".join(filename_parts[1:-2])  # All parts except first and last two
                        original_id_str = filename_parts[-2]  # Second to last part
                    else:
                        perturbation_type = filename_parts[1]
                        original_id_str = filename_parts[2]
                    try:
                        original_id = int(original_id_str)
                    except ValueError:
                        original_id = "unknown"
                else:
                    perturbation_class = "unknown"
                    perturbation_type = "unknown"
                    original_id = "unknown"
                
                part = self._parse_single_file(txt)
                for q, rec in part.items():
                    # Use original_id as the key to ensure uniqueness
                    rec["original_id"] = original_id
                    rec["perturbation_class"] = perturbation_class
                    rec["perturbation_type"] = perturbation_type
                    rec["question"] = q  # Store the original question text
                    preds[str(original_id)] = rec
                successful_parses += 1
            except Exception as exc:
                failed_parses += 1
                import warnings
                warnings.warn(f"Failed to parse {txt.name}: {exc}")
        
        print(f"Parsing summary: {successful_parses} successful, {failed_parses} failed")
        
        # convert to desired structure {qid: {...}}
        out: Dict[str, Dict[str, Any]] = {}
        for original_id, rec in preds.items():
            out[rec.get("question", f"question_{original_id}")] = {
                "answer": rec["answer"], 
                "scale": "", 
                "source": rec["source"], 
                "confidence": rec["confidence"],
                "original_id": rec.get("original_id", "unknown")
            }
        return out
    
    def consolidate(self, gold: Dict[str, Dict[str, Any]], 
                   pred: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced consolidation with perturbation metadata."""
        records = []
        
        # Create a mapping from (original_id, perturbation_class, perturbation_type) to gold entry
        gold_mapping = {}
        for g_id, g_data in gold.items():
            original_id = g_data.get("original_id", "")
            perturbation_class = g_data.get("perturbation_class", "")
            perturbation_type = g_data.get("perturbation_type", "")
            if original_id:
                key = (original_id, perturbation_class, perturbation_type)
                gold_mapping[key] = g_data
        
        print(f"Gold entries by (original_id, class, type): {len(gold_mapping)}")
        print(f"Predictions to match: {len(pred)}")
        
        # Debug: Print first few gold entries
        print("First 3 gold entries:")
        for i, ((original_id, pclass, ptype), g_data) in enumerate(gold_mapping.items()):
            if i < 3:
                print(f"  {original_id}_{pclass}_{ptype}: {g_data.get('question', 'NO_QUESTION')}")
        
        # Debug: Print first few predictions
        print("First 3 predictions:")
        for i, (question_text, p) in enumerate(pred.items()):
            if i < 3:
                print(f"  {question_text} (original_id: {p.get('original_id', 'unknown')})")
        
        # Debug: Check if the prediction original_ids exist in gold data
        pred_original_ids = set(p.get("original_id", "unknown") for p in pred.values())
        gold_original_ids = set(original_id for (original_id, pclass, ptype) in gold_mapping.keys())
        print(f"Prediction original_ids: {sorted(pred_original_ids)}")
        print(f"Gold original_ids (first 10): {sorted(list(gold_original_ids))[:10]}")
        
        for question_text, p in pred.items():
            # Get original_id from prediction
            pred_original_id = p.get("original_id", "unknown")
            
            # Try to find matching gold by original_id first (any perturbation type)
            matching_gold = None
            for (gold_original_id, pclass, ptype), g_data in gold_mapping.items():
                if gold_original_id == pred_original_id:
                    matching_gold = g_data
                    break
            
            if matching_gold:
                record = {
                    "qid": matching_gold.get("original_id", f"unknown_{len(records)}"),
                    "perturbation_class": matching_gold.get("perturbation_class", ""),
                    "perturbation_type": matching_gold.get("perturbation_type", ""),
                    "modality": matching_gold.get("modality", ""),
                    "original_id": matching_gold.get("original_id", ""),
                    "perturbation_notes": matching_gold.get("perturbation_notes", ""),
                    "question": question_text,
                    "gold_answer": matching_gold.get("answer"),
                    "pred_answer": p.get("answer"),
                    "source": p.get("source", ""),
                    "confidence": p.get("confidence", ""),
                    "answer_type": p.get("source", ""),
                }
                records.append(record)
            else:
                # If no matching gold found, create record with available data
                record = {
                    "qid": f"unknown_{len(records)}",
                    "perturbation_class": "",
                    "perturbation_type": "",
                    "modality": "",
                    "original_id": pred_original_id,
                    "perturbation_notes": "",
                    "question": question_text,
                    "gold_answer": "",
                    "pred_answer": p.get("answer"),
                    "source": p.get("source", ""),
                    "confidence": p.get("confidence", ""),
                    "answer_type": p.get("source", ""),
                }
                records.append(record)
        
        print(f"Successfully matched {len([r for r in records if r['gold_answer']])} predictions with gold data")
        return records 