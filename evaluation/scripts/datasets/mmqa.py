from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

try:
    from evaluation.scripts.base_dataset import BaseDataset  # type: ignore
except ModuleNotFoundError:
    from base_dataset import BaseDataset  # type: ignore

class MmqaDataset(BaseDataset):
    GOLD_FILE = Path(__file__).resolve().parents[3] / "data" / "mmqa" / "MMQA_dev.jsonl"

    def load_gold(self) -> Dict[str, Dict[str, Any]]:
        gold: Dict[str, Dict[str, Any]] = {}
        with self.GOLD_FILE.open("r", encoding="utf-8") as fp:
            for line in fp:
                obj = json.loads(line)
                qid = str(obj["qid"])
                # answers is list, may have dicts
                answers = obj.get("answers", [])
                if answers and isinstance(answers[0], dict):
                    answer_val = answers[0].get("answer")
                else:
                    answer_val = answers
                gold[qid] = {
                    "answer": answer_val,
                    "scale": "",
                }
        return gold 