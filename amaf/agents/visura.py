from .base import Agent
from ..core import AgentOutput


class VisuraAgent(Agent):
    def __init__(self):
        super().__init__("Visura")

    def run(self, data, logs):
        if not data.image_cues:
            return AgentOutput(cot="(no visuals)", result="")
        prompt = f"Image description: {data.image_cues}\nExplain the key takeaway in ≤2 sentences. Think first, then answer."
        cot_and_msg = self._chat(prompt)
        cot, msg = cot_and_msg.split("\n")[:2]
        return AgentOutput(cot=cot.strip(), result=msg.strip())