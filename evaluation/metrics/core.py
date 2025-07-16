"""Dataset-agnostic metric helpers: EM, token-F1 and HCS."""
from __future__ import annotations

import math
import re
from typing import Tuple, List, Set

_num_re = re.compile(r"-?\d+(?:\.\d+)?")

# Normalisation helpers
def _normalise_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _is_number(tok: str) -> bool:
    return bool(_num_re.fullmatch(tok))


def _tokens(s: str) -> List[str]:
    # basic whitespace + punctuation split
    return re.findall(r"[\w\.\-]+", s.lower())

# Exact match (after normalisation)
def exact_match(gold: str, pred: str) -> int:
    return int(_normalise_text(gold) == _normalise_text(pred))

# Token-level Precision / Recall / F1 (SQuAD style)
def token_f1(gold: str, pred: str) -> Tuple[float, float, float]:
    gold_toks = _tokens(gold)
    pred_toks = _tokens(pred)

    common = {}
    for tok in gold_toks:
        if tok in pred_toks:
            common[tok] = common.get(tok, 0) + 1
            pred_toks.remove(tok)
    num_same = sum(common.values())

    if not gold_toks and not pred_toks:
        return 1.0, 1.0, 1.0
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / max(len(gold_toks), 1)
    recall = num_same / max(len(pred_toks) + num_same, 1)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

# Hallucination Consistency Score (HCS)
def _extract_numbers(text: str) -> Set[str]:
    return set(_num_re.findall(text))


def hallucination_consistency(gold_or_source: str, summary: str) -> float:
    """HCS = 1 − (# hallucinated numbers / # numbers in summary).

    gold_or_source can be gold answer string or concatenated table text.
    """
    pred_nums = _extract_numbers(summary)
    gold_nums = _extract_numbers(gold_or_source)
    if not pred_nums:
        return 1.0  # no numbers = no hallucination
    hallucinated = pred_nums - gold_nums
    return 1.0 - (len(hallucinated) / len(pred_nums)) 

# New helpers ---------------------------------------------------------------

def answer_presence(gold: str, pred: str) -> int:
    """Return 1 if the *normalised* gold answer string occurs anywhere in *pred*.

    Both strings are canonicalised via _normalise_text before substring search so
    minor casing / whitespace differences do not affect the match.
    """
    return int(_normalise_text(gold) in _normalise_text(pred))


def answer_token_f1(gold: str, pred: str):
    """Precision/Recall/F1 computed *only* over gold answer tokens.

    We ignore extra tokens present in the prediction – the metric reflects
    coverage of the gold answer rather than penalising verbosity.
    """
    gold_toks = set(_tokens(gold))
    if not gold_toks:
        return 1.0, 1.0, 1.0  # degenerate – no gold tokens

    pred_toks = set(_tokens(pred))
    overlap = len(gold_toks & pred_toks)
    coverage = overlap / len(gold_toks)  # fraction of answer tokens found
    return coverage, coverage, coverage

# --------------------------------------------------------------------------- 