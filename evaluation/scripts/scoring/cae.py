from __future__ import annotations
"""Coverage-of-Answer Evidence metric via Gemini.

Returns 1.0 for Yes, 0.0 for No, 0.5 for unclear.
Falls back to simple substring containment when key missing or SDK unavailable.
"""

import os
import re
from typing import Any

_yes_re = re.compile(r"^\s*yes\b", re.I)
_no_re = re.compile(r"^\s*no\b", re.I)

_PROMPT_SYSTEM = (
    "You are an expert LLM evaluator tasked with assessing the accuracy of "
    "model responses against gold standard answers. Your role is to determine "
    "if the core content and intent of the model's response align with the gold answer, "
    "considering various answer formats and implicit information."
)

_TEMPLATE = (
    "Question: {question}\n"
    "Gold answer: {gold}\n\n"
    "Model response:\n{pred}\n\n"
    "Key Guidelines\n"
    "• Understand the question’s intent and any units or scale words involved.\n"
    "• Treat answers as equivalent when the numeric value matches, even if any accompanying unit / scale token (million, %, years, ft², etc.) is present only in the gold OR the prediction. Do not penalise its absence.\n"
    "• Accept shortened numeric forms (e.g., 2019 → 19, 3 000 000 → 3000) when they clearly denote the same quantity, and ignore approximation modifiers like ‘about’, ‘~’, or ‘approximately’.\n"
    "• Ignore a leading minus sign or textual qualifiers such as ‘increase’ / ‘decrease’ unless the question explicitly asks about direction.\n"
    "• A concise model response is correct if it is present in the factual content of the gold answer, even when the gold is a longer sentence.\n"
    "• For list-type answers, mark the response correct if every predicted element appears in the gold list (order irrelevant); missing additional gold elements is acceptable.\n"
    "Final Judgment\nProvide a \"Yes\" or \"No\" judgment without explanation."
)


def cae_score(*, question: str, gold: str, pred: str, gemini_key: str | None) -> float:
    # fallback simple
    if not gemini_key:
        return 1.0 if gold.lower() in pred.lower() else 0.0

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-pro")
        prompt = _TEMPLATE.format(question=question, gold=gold, pred=pred)
        resp = model.generate_content([_PROMPT_SYSTEM, prompt])
        text = resp.text.strip()
    except Exception:
        # on any SDK failure, revert to fallback
        return 1.0 if gold.lower() in pred.lower() else 0.0

    if _yes_re.match(text):
        return 1.0
    if _no_re.match(text):
        return 0.0
    return 0.5 