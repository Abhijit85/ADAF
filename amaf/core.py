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
    # For datasets with single-question inputs.
    question: str = ""
    # For datasets like TAT-QA providing multiple questions per table.
    questions: List[str] = field(default_factory=list)
    # Desired units / magnitudes per question (e.g., ["million", "%", ""])
    answer_scales: List[str] = field(default_factory=list)
    # Expected answer datatypes per question (e.g., ["number", "span", ...])
    answer_types: List[str] = field(default_factory=list)


@dataclass
class AgentOutput:
    cot: str  # chain of thought
    result: str  # agent’s main payload (facts/summary/etc.)