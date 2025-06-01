from abc import ABC, abstractmethod
from typing import Dict, Any
from ..core import InputData, AgentOutput
import openai, os

class Agent(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name
        self.model = os.getenv("AMAF_MODEL", "gpt-4")  # override via env

    def _chat(self, prompt: str, temperature: float = .0) -> str:
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    @abstractmethod
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:...