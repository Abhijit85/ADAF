from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

@dataclass
class InputData:
    table: Dict[str, Any]
    context: str = ""
    image_cues: str = ""
    user_profile: str = "general"

@dataclass
class AgentOutput:
    cot: str  # chain of thought
    result: str  # agent’s main payload (facts/summary/etc.)