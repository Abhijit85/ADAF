from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class InputData:
    """Structured input for the AMAF pipeline."""
    table: Dict[str, Any]
    context: Optional[str] = ""
    image_cues: Optional[str] = ""
    user_profile: Optional[str] = ""

class Agent(ABC):
    """Abstract base class for all AMAF agents."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, data: InputData, logs: Dict[str, Any]) -> Any:
        ...
