"""Thin wrapper around openai.ChatCompletion with retries."""
from __future__ import annotations

import os
import time
from typing import Any

import openai


def chat_complete(*, openai_key: str, system_msg: str, user_msg: str, model: str = "gpt-4o-mini", max_retries: int = 3) -> str:
    openai.api_key = openai_key
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return "" 