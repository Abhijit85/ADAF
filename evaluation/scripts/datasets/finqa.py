from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

try:
    from evaluation.scripts.base_dataset import BaseDataset  # type: ignore
except ModuleNotFoundError:
    from base_dataset import BaseDataset  # type: ignore


class FinqaDataset(BaseDataset):
    GOLD_FILE = Path(__file__).resolve().parents[3] / "data" / "finqa" / "finqa_dev.json"

    def load_gold(self) -> Dict[str, Dict[str, Any]]:
        path = self.GOLD_FILE
        with path.open("r", encoding="utf-8") as fp:
            dataset = json.load(fp)

        gold: Dict[str, Dict[str, Any]] = {}
        for item in dataset:
            qid = str(item["id"])
            qa = item["qa"]
            # Use both answer and exe_ans as potential gold answers
            answer_val = qa.get("answer")  # String answer (e.g., "2014")
            exe_ans = qa.get("exe_ans")    # Numeric answer (e.g., 155239.0)
            scale = qa.get("unit", "")
            question = qa.get("question", "")
            gold[qid] = {
                "answer": answer_val,
                "exe_ans": exe_ans,
                "scale": scale,
                "question": question,
                "id": qid
            }
        return gold

    def _parse_single_file(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """Override to parse FinQA-specific output format."""
        text = path.read_text(encoding="utf-8", errors="ignore")
        if self._FS_MARKER not in text:
            raise ValueError("marker not found")
        
        summary_part = text.split(self._FS_MARKER, 1)[1]
        # Stop at next big section heading (=== LOGS) if present
        summary_lines = []
        for line in summary_part.splitlines()[1:]:  # skip the empty line right after marker
            if line.startswith("=== "):
                break
            summary_lines.append(line)
        summary_str = "\n".join(summary_lines).strip()
        
        # Extract JSON block
        import re
        json_match = re.search(r'\{.*\}', summary_str, re.DOTALL)
        if not json_match:
            raise ValueError("Could not locate JSON block after FINAL SUMMARY")
        
        json_str = json_match.group(0)
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode error: {e}") from e

        # Extract question, answer, and source
        question = obj.get("Question", "")
        answer = obj.get("Answer", "")
        source = obj.get("Source", "")
        
        # Use question as key for matching with gold data
        return {question: {"answer": answer, "source": source}}

    def consolidate(self, gold: Dict[str, Dict[str, Any]], pred: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Override to handle question-based matching."""
        out = []
        for question, p in pred.items():
            # Find matching gold entry by question
            matching_gold = None
            for g_id, g_data in gold.items():
                if g_data.get("question") == question:
                    matching_gold = g_data
                    break
            
            if matching_gold:
                out.append({
                    "qid": matching_gold.get("id", ""),
                    "question": question,
                    "answer_type": p.get("source", ""),
                    "gold_answer": matching_gold.get("answer"),  # String answer
                    "gold_exe_ans": matching_gold.get("exe_ans"),  # Numeric answer
                    "gold_scale": matching_gold.get("scale", ""),
                    "pred_answer": p.get("answer"),
                    "source": p.get("source", ""),
                })
        return out 