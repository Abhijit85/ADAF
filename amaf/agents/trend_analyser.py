# amaf/agents/trend_analyser.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from ..core import AgentOutput, InputData
from .base import Agent

class TrendAnalyserAgent(Agent):
    """Identify and explain the two most significant numeric trends."""

    def __init__(self, dataset: str = None, provider="openai", model="gpt-3.5-turbo", method=None):
        super().__init__("TrendAnalyser", dataset, provider=provider, model=model, method=method)

    # ── main run ────────────────────────────────────────────────────────────
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # 1. Pull facts emitted by TabuSynth
        facts = logs.get("TabuSynth", {}).get("result", "").strip()
        if not facts:
            out = AgentOutput(cot="(no facts to analyse)", result="")
            logs[self.name] = out.__dict__
            return out

        # 2. Build prompt from external template
        prompt_file = self.get_prompt_path("trend_analyser.txt")
        prompt_template = prompt_file.read_text(encoding="utf-8")
        prompt = prompt_template.format(facts=facts)

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
