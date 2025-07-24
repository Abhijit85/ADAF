from __future__ import annotations

import re
from typing import Tuple

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation at ends."""
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def exact_match(gold: str, pred: str) -> int:
    g = _normalize(gold)
    p = _normalize(pred)
    if not g or not p:
        return 0
    if len(g) >= len(p):
        if p in g:
            return 1
    else:
        if g in p:
            return 1
    # fallback: alnum-only containment
    flat_g = re.sub(r"[^a-z0-9]", "", g)
    flat_p = re.sub(r"[^a-z0-9]", "", p)
    if len(flat_g) >= len(flat_p):
        return int(flat_p in flat_g)
    return int(flat_g in flat_p)


def f1_score(gold: str, pred: str) -> float:
    if exact_match(gold, pred):
        return 1.0
    g_tok = _normalize(gold).split()
    p_tok = _normalize(pred).split()
    if not g_tok or not p_tok:
        return 0.0
    common = len(set(g_tok) & set(p_tok))
    if common == 0:
        return 0.0
    precision = common / len(p_tok)
    recall = common / len(g_tok)
    return 2 * precision * recall / (precision + recall) 