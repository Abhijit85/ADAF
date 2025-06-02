from ..core import AgentOutput
from .base import Agent

class SummaCraftAgent(Agent):
    def __init__(self): super().__init__("SummaCraft")

    def run(self, data, logs):
        # Blend all previous agent results
        pieces = []
        for key in ("TabuSynth", "Contextron", "Visura", "TrendAnalyser", "TopKFilter"):
            r = logs.get(key, {}).get("result", "").strip()
            if r: pieces.append(f"**{key}**\n{r}")
        bundle = "\n\n".join(pieces)
        template = {
            "general": "Write a concise, reader‑friendly summary…",
            "retail investor": "In plain terms, explain why the numbers matter…",
            "analyst": "Assume finance expertise; focus on implications and metrics…",
        }[data.user_profile]

        prompt = f"{template}\n\nSOURCE NOTES:\n{bundle}\n\nFinal summary:""
        summary = self._chat(prompt, temperature=.2)
        return AgentOutput(cot="(omitted)", result=summary.strip())