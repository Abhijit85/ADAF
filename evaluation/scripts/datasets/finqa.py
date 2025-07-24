from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

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
            # Prefer string answer field else exe_ans numeric
            answer_val = qa.get("answer")
            if answer_val is None or answer_val == "":
                answer_val = qa.get("exe_ans")
            scale = qa.get("unit", "")
            gold[qid] = {"answer": answer_val, "scale": scale}
        return gold 