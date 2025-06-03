from .base import Agent
from ..core import AgentOutput, InputData
from typing import Dict, Any


class VisuraAgent(Agent):
    def __init__(self):
        super().__init__("Visura")

    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        if not data.image_cues:
            out = AgentOutput(cot="(no visuals)", result="")
            logs[self.name] = out.__dict__
            return out

        prompt = (
            f"Image description:\n{data.image_cues}\n\n"
            "Explain the key takeaway in ≤2 sentences. "
            "Think step-by-step, then answer."
        )
        cot_and_msg = self._chat(prompt)

        # split on first blank line OR default fallback
        parts = cot_and_msg.split("\n\n", 1)
        cot = parts[0]
        msg = parts[1] if len(parts) > 1 else cot  # if model collapsed into one block

        out = AgentOutput(cot=cot.strip(), result=msg.strip())
        logs[self.name] = out.__dict__              # ← store structured
        return out
