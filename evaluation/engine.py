# evaluation/engine.py
# Dataset-agnostic EvalEngine base class and registry helper.

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Callable, Type

from tqdm import tqdm

from .metrics.core import (
    exact_match,
    answer_presence,
    answer_token_f1,
    hallucination_consistency,
)
from .metrics.cae_gpt import cae_score

_REGISTRY: Dict[str, Type["EvalEngine"]] = {}


def register_dataset(name: str):
    """Decorator to register an evaluator class under *name*."""

    def _decorator(cls: Type["EvalEngine"]):
        _REGISTRY[name.lower()] = cls
        return cls

    return _decorator


def get_evaluator(name: str) -> Type["EvalEngine"]:
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_REGISTRY)}")


class EvalEngine(ABC):
    """Abstract base: child provides _load_gold() and _iter_predictions()."""

    def __init__(
        self,
        gold_path: Path,
        pred_path: Path,
        mode: str = "summary+echoes",
        pred_kind: str = "summary",
        openai_key: str | None = None,
    ) -> None:
        self.gold_path = gold_path
        self.pred_path = pred_path
        self.mode = mode
        self.pred_kind = pred_kind  # "summary" or "answer"
        self.openai_key = openai_key

        self.gold_dict: Dict[str, str] = self._load_gold()

    @abstractmethod
    def _load_gold(self) -> Dict[str, str]:
        """Return mapping {qid: canonicalised_gold_answer_str}."""

    @abstractmethod
    def _iter_predictions(self):
        """Yield tuples (qid, raw_prediction_text, metadata_dict)."""

    def evaluate(self) -> Dict[str, Any]:
        per_q: List[Dict[str, Any]] = []

        for qid, pred_raw, meta in tqdm(list(self._iter_predictions()), desc="scoring"):
            gold = self.gold_dict.get(qid)
            if gold is None:
                # silently skip – unknown id
                continue

            row: Dict[str, Any] = {"qid": qid, **meta, "gold": gold, "pred": pred_raw}

            # core metrics
            row["em"] = exact_match(gold, pred_raw)
            row["contains"] = answer_presence(gold, pred_raw)
            pr, rc, f1 = answer_token_f1(gold, pred_raw)
            row["p"], row["r"], row["f1"] = pr, rc, f1
            # For our use-case REMS mirrors the modified F1 metric
            row["rems"] = f1

            if self.pred_kind == "summary":
                # summary-specific metrics (HCS & CAE retained)
                row["hcs"] = hallucination_consistency(gold, pred_raw)
                row["cae"] = cae_score(
                    question=meta.get("question", ""),
                    gold_answer=gold,
                    summary=pred_raw,
                    openai_key=self.openai_key,
                )
            per_q.append(row)

        # aggregate
        agg: Dict[str, Any] = {k: None for k in ("em", "contains", "f1", "rems", "hcs", "cae")}
        n = len(per_q)
        if n == 0:
            return {"error": "no predictions found", "per_question": per_q}

        for metric in agg:
            vals = [r[metric] for r in per_q if metric in r and r[metric] is not None]
            if vals:
                agg[metric] = sum(vals) / len(vals)
        agg["count"] = n
        agg["per_question"] = per_q
        return agg 