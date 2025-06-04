# amaf/agents/trend_analyser.py

from __future__ import annotations

import re
from typing import Any, Dict

from ..core import AgentOutput, InputData
from .base import Agent


class TrendAnalyserAgent(Agent):
    """Identify and explain the two most significant numeric trends."""

    def __init__(self) -> None:
        super().__init__("TrendAnalyser")

    # ── main run ────────────────────────────────────────────────────────────
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # 1. Pull facts emitted by TabuSynth
        facts = logs.get("TabuSynth", {}).get("result", "").strip()
        if not facts:
            out = AgentOutput(cot="(no facts to analyse)", result="")
            logs[self.name] = out.__dict__
            return out

        # 2. Build prompt
        prompt = (
            f"FACTS (verbatim from table):\n{facts}\n\n"
            "You are a financial analyst asked to surface the *two* numeric trends\n"
            "that matter most for decision‑makers.  A *trend* is a notable increase\n"
            "or decrease implied by the numbers (e.g., 2‑year change, YoY delta,\n"
            "ratio shift).\n\n"
            "### OUTPUT LAYOUT (strict)\n"
            "<one short reasoning paragraph>\n\n"
            "<exactly 2 bullets, each starting with ▲ or ▼, quoting numbers>\n\n"
            "### RULES\n"
            "1. Compute % change when a fact shows A→B or A/B.\n"
            "2. Quote all numbers & units from the facts; do *not* invent.\n"
            "3. Bullet ≤ 25 words.\n"
            "4. Think between ##### markers; that text will be removed.\n\n"
            "##### INTERNAL SCRATCHPAD (think here)\n#####\n\n"
            "Now generate the final answer:"
        )

        cot_and_bullets = self._chat(prompt, temperature=0.25)

        # 3. Strip scratchpad if echoed back
        cot_and_bullets = re.sub(r"#####.*?#####", "", cot_and_bullets, flags=re.DOTALL).strip()

        # 4. Separate reasoning and bullets
        parts = cot_and_bullets.split("\n\n", 1)
        cot, bullet_block = parts if len(parts) == 2 else ("", cot_and_bullets)

        out = AgentOutput(cot=cot.strip(), result=bullet_block.strip())

        # 5. Log raw output for debugging
        logs[self.name] = out.__dict__ | {"raw": cot_and_bullets}
        return out
