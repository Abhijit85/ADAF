# amaf/agents/summa_craft.py 

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict

from ..core import AgentOutput, InputData
from .base import Agent


# ───────── helpers ─────────────────────────────────────────────────────
def canonicalise_numbers(text: str) -> str:
    """Remove thin spaces / NBSP and digit-group commas (60,000 → 60000)."""
    text = re.sub(r"[\u202F\u00A0]", "", text)              # thin / nbsp
    return re.sub(r"(?<=\d),(?=\d)", "", text)              # commas


# ───────── main agent ─────────────────────────────────────────────────
class SummaCraftAgent(Agent):
    """Synthesize upstream notes into a concise, audience-aware summary."""

    def __init__(self) -> None:
        super().__init__("SummaCraft")

    # ------------------------------------------------------------------
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # 1. Collect upstream notes (TabuSynth, Contextron, …)
        raw_blocks: list[str] = []
        for key in ("TabuSynth", "Contextron", "Visura",
                    "TrendAnalyser", "TopKFilter"):
            txt = logs.get(key, {}).get("result", "")
            txt = re.sub(r"^\[\w+] *", "", txt).strip()          # drop “[Agent]”
            if txt:
                raw_blocks.append(txt)

        bundle_for_reasoning = "\n\n".join(raw_blocks) or "(no notes)"

        # Also gather individual bullet lines for Answer Echoes
        echo_lines = [
            ln.strip() for blk in raw_blocks
            for ln in blk.splitlines()
            if ln.strip().lstrip().startswith("-")
        ]

        # 2. Audience-specific template
        templates = {
            "general": (
                "Write a concise analysis ≈150 words: start with one-sentence "
                "overview, then 3–4 numbered bullets on key quantitative and "
                "contextual insights."
            ),
            "retail investor": (
                "Explain in plain language why these figures matter to a non-expert "
                "investor. Use ≤180 words, same structure."
            ),
            "analyst": (
                "Provide an expert synthesis focusing on material trends, anomalies, "
                "and valuation drivers. ≤200 words."
            ),
        }
        template = templates.get(data.user_profile, templates["general"])

        # 3. Build prompt with INTERNAL scratchpad
        prompt = textwrap.dedent(f"""
            {template}

            SOURCE NOTES (verbatim):
            {bundle_for_reasoning}

            TASK
            ----
            1. ##### Think in detail here to pick 3-5 decision-relevant points
               and craft a narrative. This block will be removed. #####
            2. Output ONLY the final public summary:
               • one-sentence overview
               • 3–4 numbered bullets
            3. Quote every number exactly as in the notes; do NOT invent data.

            Begin now:
        """)

        # 4. Call LLM and post-process
        raw_resp = self._chat(prompt, temperature=0).strip()
        # strip scratchpad if the model echoed it
        cleaned_resp = re.sub(r"#####.*?#####", "", raw_resp, flags=re.DOTALL).strip()

        summary = canonicalise_numbers(cleaned_resp)

        # 5. Build Answer Echoes
        answer_echoes = "\n".join(echo_lines) or "(none)"

        final_text = f"{summary.rstrip()}\n\nAnswer Echoes:\n{answer_echoes}"

        # 6. Package result
        out = AgentOutput(cot="(omitted)", result=final_text)
        logs[self.name] = out.__dict__ | {"raw": raw_resp}
        return out
