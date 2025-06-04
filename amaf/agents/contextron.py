# amaf/agents/contextron.py

from __future__ import annotations

import re
from typing import Any, Dict

from ..core import AgentOutput, InputData
from .base import Agent


class ContextronAgent(Agent):
    """Summarise free‑form context into tagged insights."""

    TAG_SET = "[DEFINITION] [SCOPE] [SOURCE] [NOTE] [WARNING]"

    def __init__(self) -> None:
        super().__init__("Contextron")

    # ── main run ────────────────────────────────────────────────────────────
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # 1. No context? early‑return
        if not data.context:
            out = AgentOutput(cot="(no context)", result="")
            logs[self.name] = out.__dict__
            return out

        # 2. Build prompt
        prompt = (
            f"CONTEXT (verbatim):\n{data.context}\n\n"
            "You are an analyst helping auditors interpret the above context in"
            " relation to a financial table.  Extract only details that *directly"
            " clarify* numbers, columns, time ranges, or caveats in the table.\n\n"
            "### OUTPUT LAYOUT (strict)\n"
            "<one concise reasoning paragraph>\n"
            "\n"  # single blank line
            "<3‑7 bullets, each starting with exactly one tag from"
            f" {self.TAG_SET}>\n\n"
            "### RULES\n"
            "1. Never invent data not present in the context.\n"
            "2. Quote important phrases in \"double quotes\".\n"
            "3. Keep bullet text ≤20 words.\n"
            "4. You may think between ##### markers; that text will be removed.\n\n"
            "##### INTERNAL SCRATCHPAD (think here)\n#####\n\n"
            "Now generate the final answer:"
        )

        cot_and_ins = self._chat(prompt, temperature=0.25)

        # 3. Remove any internal scratchpad echoed back
        cot_and_ins = re.sub(r"#####.*?#####", "", cot_and_ins, flags=re.DOTALL).strip()

        # 4. Separate CoT and insights
        parts = cot_and_ins.split("\n\n", 1)
        if len(parts) == 2:
            cot, insight_block = parts
        else:
            cot, insight_block = "", cot_and_ins  # model didn't insert blank line

        out = AgentOutput(cot=cot.strip(), result=insight_block.strip())

        # 5. Log raw output for debugging
        logs[self.name] = out.__dict__ | {"raw": cot_and_ins}
        return out
