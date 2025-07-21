from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class InputData:
    table: Dict[str, Any]
    context: str = ""
    image_cues: str = ""
    image_path: List[str] = field(default_factory=list)
    user_profile: str = "general"
    questions: Dict[str, str] = field(default_factory=dict)  # qid -> question mapping for TATQA


@dataclass
class AgentOutput:
    cot: str  # chain of thought
    result: str  # agent's main payload (facts/summary/etc.)