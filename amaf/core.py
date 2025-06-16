from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class InputData:
    table: Dict[str, Any]
    context: str = ""
    image_cues: str = ""
    user_profile: str = "general"
    questions: str = ""


@dataclass
class AgentOutput:
    cot: str  # chain of thought
    result: str  # agent's main payload (facts/summary/etc.)