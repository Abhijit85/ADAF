from ..core import InputData, AgentOutput
from .base import Agent

class ContextronAgent(Agent):
    def __init__(self):
        super().__init__("Contextron")

    def run(self, data, logs):
        if not data.context:
            return AgentOutput(cot="(no context)", result="")
        prompt = f"Context:\n{data.context}\n\nSummarise only info that clarifies the table above. Think step‑by‑step then list insights.""
        cot_and_ins = self._chat(prompt)
        cot, *ins = cot_and_ins.split("\n\n")
        return AgentOutput(cot=cot.strip(), result="\n".join(ins).strip())