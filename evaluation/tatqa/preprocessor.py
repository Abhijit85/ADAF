"""TAT-QA answer normalisation utilities.

Ported & simplified from original tatqa_metric.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

_scale_words = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "percent": 0.01, "": 1}
_num_re = re.compile(r"-?\d+(?:\.\d+)?")


def _clean_text(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()


def _normalise(answer: str | list, scale: str | None = "") -> str:
    """Canonical answer string: join list by space, apply scale word."""
    if isinstance(answer, list):
        answer = " ".join(map(str, answer))
    answer = str(answer).strip()
    if scale and scale.lower() not in ("", "none"):
        answer = f"{answer} {scale.lower()}"
    return _clean_text(answer)


def load_gold(path: Path) -> Dict[str, str]:
    """Return mapping {question_id: canonical_answer}."""
    data = json.loads(path.read_text())
    gold: Dict[str, str] = {}
    for ex in data:
        for qa in ex.get("questions", []):
            qid = qa.get("uid") or qa.get("id")
            ans = qa.get("answer")
            scale = qa.get("scale", "")
            gold[qid] = _normalise(ans, scale)
    return gold

__all__ = ["load_gold", "_normalise"] 