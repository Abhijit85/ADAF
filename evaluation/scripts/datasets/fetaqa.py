from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List
import re
import warnings

try:
    from evaluation.scripts.base_dataset import BaseDataset  # type: ignore
except ModuleNotFoundError:
    from base_dataset import BaseDataset  # type: ignore

class FetaqaDataset(BaseDataset):
    GOLD_FILE = Path(__file__).resolve().parents[3] / "data" / "FETAQA" / "fetaQA-v1_dev.jsonl"

    # ------------------------------------------------------------------
    def load_gold(self) -> Dict[str, Dict[str, Any]]:
        """Return mapping question_text -> {feta_id, answer}"""
        gold: Dict[str, Dict[str, Any]] = {}
        path = self.GOLD_FILE
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                obj = json.loads(line)
                q_text = str(obj.get("question", "")).strip()
                gold[q_text] = {
                    "feta_id": str(obj["feta_id"]),
                    "answer": obj.get("answer", ""),
                }
        return gold

    # ------------------------------------------------------------------
    # prediction parsing (raw *_out.txt files)
    # ------------------------------------------------------------------

    _FS_MARKER = "=== FINAL SUMMARY ==="

    def _parse_single_file(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """Return mapping question_str -> {answer, source, confidence}."""
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

    # ------------------------------------------------------------------
    def parse_run_dir(self, run_dir: str | Path) -> Dict[str, Dict[str, Any]]:  # type: ignore[override]
        run_path = Path(run_dir)
        if not run_path.is_dir():
            raise FileNotFoundError(run_dir)

        preds: Dict[str, Dict[str, Any]] = {}
        dupes: set[str] = set()
        for txt in sorted(run_path.glob("*_out.txt")):
            try:
                part = self._parse_single_file(txt)
                for q, rec in part.items():
                    if q in dupes:
                        continue
                    if q in preds:
                        # mark duplicate – drop
                        preds.pop(q, None)
                        dupes.add(q)
                    else:
                        preds[q] = rec
            except Exception as exc:
                warnings.warn(f"Failed to parse {txt.name}: {exc}")
        # convert to desired structure {qid: {...}}
        out: Dict[str, Dict[str, Any]] = {}
        for q, rec in preds.items():
            out[q] = {"answer": rec["answer"], "scale": "", "source": rec["source"], "confidence": rec["confidence"]}
        return out

    # ------------------------------------------------------------------
    def consolidate(self, gold: Dict[str, Dict[str, Any]], pred: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:  # type: ignore[override]
        records: List[Dict[str, Any]] = []
        for question_text, p in pred.items():
            g = gold.get(question_text, {})
            records.append({
                "qid": g.get("feta_id", "UNKNOWN"),
                "feta_id": g.get("feta_id", "UNKNOWN"),
                "question": question_text,
                "gold_answer": g.get("answer"),
                "pred_answer": p.get("answer"),
                "source": p.get("source", ""),
                "confidence": p.get("confidence", ""),
                # use source as answer_type for grouping
                "answer_type": p.get("source", ""),
            })
        return records 