from __future__ import annotations

"""Shared utilities for gold/pred parsing and consolidation."""

import json
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

__all__ = ["BaseDataset"]


class BaseDataset:
    """Dataset-specific subclasses must provide load_gold()."""

    # Path to gold file – each subclass sets this
    GOLD_FILE: Path | str | None = None

    def load_gold(self) -> Dict[str, Dict[str, Any]]:
        """Return mapping qid -> {answer, scale}.  Must be implemented by subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Generic helpers – identical across datasets
    # ------------------------------------------------------------------
    def parse_run_dir(self, run_dir: str | Path) -> Dict[str, Dict[str, Any]]:
        """Parse every *_out.txt inside a run folder.

        Expected format inside each file::

            ...
            === FINAL SUMMARY ===\n
            {"qid": [answer, scale], ...}
            (blank line OR "=== LOGS" marker)
        """
        run_path = Path(run_dir)
        if not run_path.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_path}")

        preds: Dict[str, Dict[str, Any]] = {}

        for txt_file in sorted(run_path.glob("*_out.txt")):
            try:
                new_preds = self._parse_single_file(txt_file)
                # merge with overwrite detection
                for qid, record in new_preds.items():
                    if qid in preds:
                        warnings.warn(
                            f"Duplicate prediction for qid {qid} in {txt_file.name}; overwriting."
                        )
                    preds[qid] = record
            except Exception as exc:  # broad – better to keep going
                warnings.warn(f"Failed to parse {txt_file.name}: {exc}")
        return preds

    # -------------------------- helpers -------------------------------
    _FS_MARKER = "=== FINAL SUMMARY ==="

    def _parse_single_file(self, path: Path) -> Dict[str, Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if self._FS_MARKER not in text:
            raise ValueError("marker not found")
        summary_part = text.split(self._FS_MARKER, 1)[1]
        # Stop at next big section heading (=== LOGS) if present
        summary_lines: List[str] = []
        for line in summary_part.splitlines()[1:]:  # skip the empty line right after marker
            if line.startswith("=== "):
                break
            summary_lines.append(line)
        summary_str = "\n".join(summary_lines).strip()
        # Some agents wrap JSON in ```json ... ```
        cleaned = summary_str.strip()
        cleaned = re.sub(r"```json|```", "", cleaned).strip()

        summary_json_match = re.search(r"\{.*\}\s*$", cleaned, re.DOTALL)
        if not summary_json_match:
            raise ValueError("Could not locate JSON block after FINAL SUMMARY")
        json_str = summary_json_match.group(0)
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode error: {e}") from e

        preds: Dict[str, Dict[str, Any]] = {}
        for qid, pair in obj.items():
            if isinstance(pair, dict):
                answer = pair.get("answer")
                scale = pair.get("scale", "")
            elif isinstance(pair, list):
                # Heuristic: if second element looks like a pure scale token
                if len(pair) == 2 and self._looks_like_scale(pair[1]):
                    answer, scale = pair[0], pair[1]
                else:
                    answer, scale = pair, ""
            else:
                answer, scale = pair, ""
            preds[str(qid)] = {"answer": answer, "scale": scale}
        return preds

    # ------------------------------------------------------------------
    _KNOWN_SCALES = {"", "million", "thousand", "percent", "billion", "dollars", "eur", "euro", "kgs", "kg"}

    @classmethod
    def _looks_like_scale(cls, s: Any) -> bool:
        """Return True if the string resembles a unit/scale token and not an answer value."""
        if not isinstance(s, str):
            return False
        token = s.strip().lower()
        # Reject if token contains a digit or punctuation other than percent sign char inside (shouldn't) but we treat numbers as not scale
        if any(ch.isdigit() for ch in token):
            return False
        # Remove trailing % sign for check
        token = token.rstrip("%")
        return token in cls._KNOWN_SCALES

    # ------------------------------------------------------------------
    @staticmethod
    def consolidate(gold: Dict[str, Dict[str, Any]],
                    pred: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return list of unified records ready to dump."""
        # Only include QIDs that actually have predictions; the evaluation
        # benchmark expects exactly the set produced by the run.
        out: List[Dict[str, Any]] = []
        for qid in sorted(pred):
            g = gold.get(qid, {})
            p = pred.get(qid, {})
            out.append({
                "qid": qid,
                "answer_type": g.get("answer_type", ""),
                "gold_answer": g.get("answer"),
                "gold_scale": g.get("scale", ""),
                "pred_answer": p.get("answer"),
                "pred_scale": p.get("scale", ""),
            })
        return out

    # ------------------------------------------------------------------
    def dump(self, records: List[Dict[str, Any]], dataset: str, run_dir: str | Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder_name = Path(run_dir).name
        target_dir = Path(__file__).resolve().parent.parent / dataset / run_folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"consolidated_{ts}.json"
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(records, fp, indent=2, ensure_ascii=False)
        return out_path 