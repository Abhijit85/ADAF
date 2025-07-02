# evaluation/tatqa/scorer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple
from ast import literal_eval
import json as _json

from ..engine import EvalEngine, register_dataset
from .preprocessor import load_gold, _normalise


@register_dataset("tatqa")
class TatqaEvaluator(EvalEngine):
    def _load_gold(self):
        data = _json.loads(self.gold_path.read_text())
        self._q_text: Dict[str, str] = {}
        answers: Dict[str, str] = {}
        for ex in data:
            for qa in ex.get("questions", []):
                qid = qa.get("uid") or qa.get("id")
                self._q_text[qid] = qa.get("question", "")
                answers[qid] = _normalise(qa.get("answer"), qa.get("scale", ""))
        return answers

    def _iter_predictions(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """Yield (qid, summary_text, meta). Accept dir or JSON mapping."""
        p = self.pred_path
        if p.is_dir():
            for file in p.glob("*_out.txt"):
                qid = file.stem.replace("_out", "")
                raw = file.read_text()
                summary_part, logs_text = raw, ""
                if "=== FINAL SUMMARY ===" in raw and "=== LOGS" in raw:
                    summary_part = raw.split("=== LOGS", 1)[0]
                    logs_text = raw.split("=== LOGS", 1)[1]
                summary = self._extract_summary(summary_part)
                agent_cols: Dict[str, str] = {}
                if logs_text:
                    import ast
                    try:
                        logs_dict = ast.literal_eval(logs_text.strip())
                        for agent, payload in logs_dict.items():
                            agent_cols[f"agent_{agent}"] = payload.get("result", "")
                    except Exception:
                        pass
                answer_echoes = ""
                if "Answer Echoes:" in summary:
                    answer_echoes = summary.split("Answer Echoes:", 1)[1].strip()
                meta = {
                    "question": self._q_text.get(qid, ""),
                    "answer_echoes": answer_echoes,
                }
                meta.update(agent_cols)
                yield qid, summary, meta
        else:
            import json

            data = json.loads(p.read_text())
            for qid, summary in data.items():
                yield qid, str(summary), {}

    def _extract_summary(self, text: str) -> str:
        if "=== FINAL SUMMARY ===" in text:
            text = text.split("=== FINAL SUMMARY ===", 1)[1]
        if "=== LOGS" in text:
            text = text.split("=== LOGS", 1)[0]
        if self.mode == "summary":
            idx = text.find("Answer Echoes:")
            if idx != -1:
                text = text[:idx]
        return text.strip() 