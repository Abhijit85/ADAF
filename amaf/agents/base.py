from abc import ABC, abstractmethod
from typing import Dict, Any
from ..core import InputData, AgentOutput
import openai
import os


class Agent(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name
        # override default model via env or fallback to valid default
        default_model = "gpt-4"
        self.model = os.getenv("AMAF_MODEL", default_model)
    # ensure the OPENAI_API_KEY is set in the environment

    def _chat(self, prompt: str, temperature: float = .0) -> str:
        resp = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    @abstractmethod
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        pass
    