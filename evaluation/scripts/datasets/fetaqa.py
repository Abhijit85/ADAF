from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

try:
    from evaluation.scripts.base_dataset import BaseDataset  # type: ignore
except ModuleNotFoundError:
    from base_dataset import BaseDataset  # type: ignore

class FetaqaDataset(BaseDataset):
    GOLD_FILE = Path(__file__).resolve().parents[3] / "data" / "FETAQA" / "fetaQA-v1_dev.jsonl"

    def load_gold(self) -> Dict[str, Dict[str, Any]]:
        gold: Dict[str, Dict[str, Any]] = {}
        path = self.GOLD_FILE
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                obj = json.loads(line)
                qid = str(obj["feta_id"])
                gold[qid] = {
                    "answer": obj.get("answer"),
                    "scale": "",
                }
        return gold 