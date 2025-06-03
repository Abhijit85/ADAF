from typing import Dict, Any
from ..core import AgentOutput, InputData
from .base import Agent


class TrendAnalyserAgent(Agent):
    def __init__(self):
        super().__init__("TrendAnalyser")

    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # Safeguard: TabuSynth might be missing (e.g. earlier error)
        facts = logs.get("TabuSynth", {}).get("result", "").strip()
        if not facts:
            out = AgentOutput(cot="(no facts to analyse)", result="")
            logs[self.name] = out.__dict__
            return out

        prompt = (
            f"Facts extracted from the table:\n{facts}\n\n"
            "Identify the two most significant numeric trends and explain why they matter."
        )
        resp = self._chat(prompt, temperature=0.2)   # slight creativity helps
        out = AgentOutput(cot="(omitted)", result=resp.strip())
        logs[self.name] = out.__dict__               # record for downstream use
        return out
