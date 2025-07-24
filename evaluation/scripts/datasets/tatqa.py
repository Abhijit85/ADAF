from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

# Support both package and script execution
try:
    from evaluation.scripts.base_dataset import BaseDataset  # type: ignore
except ModuleNotFoundError:
    from base_dataset import BaseDataset  # type: ignore


class TatqaDataset(BaseDataset):
    """Gold loader for TatQA dev set."""

    GOLD_FILE = Path(__file__).resolve().parents[3] / "data" / "tatqa" / "tatqa_dataset_dev.json"

    def load_gold(self) -> Dict[str, Dict[str, Any]]:
        path = self.GOLD_FILE
        with path.open("r", encoding="utf-8") as fp:
            dataset = json.load(fp)

        gold: Dict[str, Dict[str, Any]] = {}

        # Dataset is an array of records; each has "questions" list
        if isinstance(dataset, list):
            items = dataset
        else:
            items = [dataset]

        for item in items:
            for q in item.get("questions", []):
                qid = str(q["uid"])
                gold[qid] = {
                    "answer": q.get("answer"),
                    "scale": q.get("scale", ""),
                    "answer_type": q.get("answer_type", ""),
                }
                self._atype_map[qid] = q.get("answer_type", "")
        return gold

    def __init__(self):
        # build map once
        self._atype_map: Dict[str, str] = {}
        # populate via load_gold
        self._gold_cache = None

    @property
    def answer_type_map(self) -> Dict[str, str]:
        if not self._atype_map:
            _ = self.load_gold()
        return self._atype_map 