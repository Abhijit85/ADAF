# amaf/agents/summa_craft.py
from typing import Dict, Any
import re, textwrap
from ..core import AgentOutput, InputData
from .base import Agent


def canonicalise_numbers(txt: str) -> str:
    """60,000 → 60000  | thin-space → '' """
    txt = re.sub(r"[\u202F\u00A0]", "", txt)          # thin / nbsp
    return re.sub(r"(?<=\d),(?=\d)", "", txt)         # commas in digits


class SummaCraftAgent(Agent):
    def __init__(self):
        super().__init__("SummaCraft")

    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # -------- harvest upstream notes ---------------------------------
        raw_blocks = []
        for key in ("TabuSynth", "Contextron", "Visura",
                    "TrendAnalyser", "TopKFilter"):
            txt = logs.get(key, {}).get("result", "")
            txt = re.sub(r"^\[\w+] *", "", txt).strip()    # drop [Agent] tag
            if txt:
                raw_blocks.append(txt)

        # flat list of individual lines (bullets, insights, etc.)
        all_lines = [ln.strip()
                     for block in raw_blocks
                     for ln in block.splitlines()
                     if ln.strip()]

        bundle_for_reasoning = "\n\n".join(raw_blocks) or "(no notes)"

        # -------- profile-aware template ---------------------------------
        templates = {
            "general": (
                "Write a concise analysis (≈150 words). Start with a one-"
                "sentence overview, then provide 3-4 short bullets of key "
                "quantitative & contextual insights."
            ),
            "retail investor": (
                "Explain in plain language why these numbers matter for a "
                "non-expert investor. ≤180 words."
            ),
            "analyst": (
                "Provide an expert-level synthesis highlighting material "
                "trends, anomalies, and valuation implications. ≤200 words."
            ),
        }
        template = templates.get(data.user_profile, templates["general"])

        # -------- prompt (NO Echo step; we add it in code) ---------------
        prompt = textwrap.dedent(f"""
            {template}

            SOURCE NOTES (verbatim):
            {bundle_for_reasoning}

            TASK:
            1. Think step-by-step to identify the 3–5 most decision-relevant points.
            2. Draft the final summary with:
               • One-sentence overview
               • 3–4 numbered bullet points

            Output ONLY the final summary prose (no Answer-Echo section).
        """)

        summary = canonicalise_numbers(
            self._chat(prompt, temperature=0).strip()
        )

        # -------- build Answer Echoes programmatically -------------------
        echo_lines = [ln for ln in all_lines if ln.lstrip().startswith("-")]
        answer_echoes = "\n".join(echo_lines) or "(none)"

        final_text = (
            f"{summary.rstrip()}\n\n"
            "Answer Echoes:\n"
            f"{answer_echoes}"
        )

        out = AgentOutput(cot="(omitted)", result=final_text)
        logs[self.name] = out.__dict__
        return out
