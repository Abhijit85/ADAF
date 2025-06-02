from ..core import AgentOutput
from .base import Agent


class TrendAnalyserAgent(Agent):
    def __init__(self): super().__init__("TrendAnalyser")

    def run(self, data, logs):
        src = logs["TabuSynth"]["result"]
        prompt = f"Given these facts:\n{src}\nIdentify which two trends are most significant and why."
        resp = self._chat(prompt)
        return AgentOutput(cot="n/a", result=resp)