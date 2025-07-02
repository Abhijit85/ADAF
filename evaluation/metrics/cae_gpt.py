"""Coverage-of-Answer Evidence metric graded by GPT.

Returns 1 for "Yes" and 0 for "No" as per rubric.
If no openai_key provided, falls back to simple string containment.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from ..openai_helper import chat_complete

_yes_re = re.compile(r"^\s*yes\b", re.I)
_no_re = re.compile(r"^\s*no\b", re.I)


@lru_cache(maxsize=128)
def cae_score(
    *,
    question: str,
    gold_answer: str,
    summary: str,
    openai_key: str | None = None,
) -> float:
    """Return 1 or 0 based on GPT rubric (or simple check if key missing)."""

    # Fallback simple string containment when key not available
    if not openai_key:
        return float(gold_answer.lower() in summary.lower())

    prompt_system = (
        "You are an expert LLM evaluator tasked with assessing the accuracy of "
        "model responses against gold standard answers. Your role is to determine "
        "if the core content and intent of the model's response align with the gold answer, "
        "considering various answer formats and implicit information."
    )

    prompt_user = f"""Question: {question}\nGold answer: {gold_answer}\n\nModel response:\n{summary}\n\nKey Guidelines\n• Understand the question's essence, including specific operations or units mentioned.\n• Compare model responses to gold answers, focusing on key information.\n• Allow a small margin of error (±0.1%) for numerical answers.\n• Recognize correct answers in different formats, such as percentages and decimals.\n• Consider implicit information and context in responses.\n• For list-type answers:\n  – Evaluate based on content rather than order.\n  – If more than two elements are missing, evaluate as incorrect.\n• Assess mathematical answers based on value range unless a specific value is required.\n• Check for appropriate units in mathematical answers.\n\nFinal Judgment\nProvide a \"Yes\" or \"No\" judgment without explanation."""

    resp = chat_complete(
        openai_key=openai_key,
        system_msg=prompt_system,
        user_msg=prompt_user,
    )
    if _yes_re.match(resp):
        return 1.0
    if _no_re.match(resp):
        return 0.0
    # unexpected answer → partial credit 0.5
    return 0.5 