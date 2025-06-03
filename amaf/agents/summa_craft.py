from typing import Dict, Any
from ..core import AgentOutput, InputData
from .base import Agent


class SummaCraftAgent(Agent):
    def __init__(self):
        super().__init__("SummaCraft")

    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # ---------- Collect upstream insights ----------
        pieces = []
        for key in ("TabuSynth", "Contextron", "Visura",
                    "TrendAnalyser", "TopKFilter"):
            txt = logs.get(key, {}).get("result", "").strip()
            if txt:
                pieces.append(f"[{key}]\n{txt}")

        bundle = "\n\n".join(pieces) or "(no additional notes)"

        # ---------- Audience-specific framing ----------
        templates = {
            "general": (
                "Write a concise summary (120-180 words) that stitches "
                "together the key *quantitative* insights **and** any critical "
                "contextual clarifications. Start with a one-sentence overview "
                "then give 3–4 bullet points."
            ),
            "retail investor": (
                "Explain in plain language why the numbers matter. Emphasise "
                "trends, risks, or opportunities in ≤180 words."
            ),
            "analyst": (
                "Provide an expert-level synthesis: highlight material "
                "trends, anomalies, and likely implications for valuation or "
                "operational KPIs. ≤200 words."
            ),
        }
        template = templates.get(data.user_profile, templates["general"])

        # ---------- Prompt ----------
        prompt = (
            f"{template}\n\n"
            "SOURCE NOTES (verbatim, grouped by agent):\n"
            f"{bundle}\n\n"
            "STEP-BY-STEP REASONING:\n"
            "1. Identify the 3–5 most decision-relevant facts from the source notes.\n"
            "2. Map each fact to its significance.\n"
            "3. Draft the final summary as requested.\n"
            "====\n"
            "Now output ONLY the final summary."
        )

        summary = self._chat(prompt, temperature=0).strip()

        out = AgentOutput(cot="(omitted)", result=summary)
        logs[self.name] = out.__dict__
        return out
