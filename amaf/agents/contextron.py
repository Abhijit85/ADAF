from typing import Dict, Any
from ..core import AgentOutput, InputData
from .base import Agent


class ContextronAgent(Agent):
    def __init__(self):
        super().__init__("Contextron")

    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        if not data.context:
            out = AgentOutput(cot="(no context)", result="")
            logs[self.name] = out.__dict__
            return out

        prompt = (
            f"Context:\n{data.context}\n\n"
            "Summarise only information that clarifies the table above. "
            "Think step-by-step, then list insights."
        )
        # zero-temperature call (default in _chat)
        cot_and_ins = self._chat(prompt)

        # first blank-line split ⇒ cot + insights
        cot, *ins = cot_and_ins.split("\n\n", 1)
        out = AgentOutput(
            cot=cot.strip(),
            result=(ins[0].strip() if ins else "")
        )
        logs[self.name] = out.__dict__
        return out
