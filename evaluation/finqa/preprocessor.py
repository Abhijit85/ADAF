from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

_num_re = re.compile(r"-?\d+(?:\.\d+)?")


def _clean(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()


def _normalise(ans) -> str:
    if isinstance(ans, list):
        ans = " ".join(map(str, ans))
    return _clean(ans)


def load_gold(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text())
    gold: Dict[str, str] = {}
    for ex in data:
        for qa in ex.get("questions", ex.get("qa_pairs", [])):
            qid = qa.get("id") or qa.get("uid") or qa.get("question_id")
            ans = qa.get("answer")
            gold[qid] = _normalise(ans)
    return gold

__all__ = ["load_gold", "_normalise"] 