"""Relaxed Exact Match Score (REMS).

F1 overlap of tokens with special ±5 % tolerance on numeric tokens.
"""
from __future__ import annotations

import math
import re
from typing import List, Tuple

_num_re = re.compile(r"-?\d+(?:\.\d+)?")


def _tokens(s: str) -> List[str]:
    return re.findall(r"[\w\.\-]+", s.lower())


def _num_close(a: str, b: str, tol: float = 0.05) -> bool:
    try:
        fa, fb = float(a), float(b)
        if fa == 0:
            return abs(fb) < tol
        return abs(fa - fb) / abs(fa) <= tol
    except ValueError:
        return False


def _match_token(pred_tok: str, gold_tok: str) -> bool:
    if _num_re.fullmatch(pred_tok) and _num_re.fullmatch(gold_tok):
        return _num_close(pred_tok, gold_tok)
    return pred_tok == gold_tok


def rems_f1(gold: str, pred: str) -> float:
    gold_toks = _tokens(gold)
    pred_toks = _tokens(pred)

    matched = 0
    unmatched_pred = pred_toks.copy()

    for g in gold_toks:
        for p in list(unmatched_pred):
            if _match_token(p, g):
                matched += 1
                unmatched_pred.remove(p)
                break

    if matched == 0:
        return 0.0
    precision = matched / len(pred_toks) if pred_toks else 0.0
    recall = matched / len(gold_toks) if gold_toks else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall) 