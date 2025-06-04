# amaf/agents/visura.py

from __future__ import annotations

import re
from typing import Any, Dict

from ..core import AgentOutput, InputData
from .base import Agent


class VisuraAgent(Agent):
    """Summarise `image_cues` into a two-sentence insight."""

    def __init__(self) -> None:
        super().__init__("Visura")

    # ── main run ────────────────────────────────────────────────────────
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # 1. Early exit if no visual cues
        if not data.image_cues:
            out = AgentOutput(cot="(no visuals)", result="")
            logs[self.name] = out.__dict__
            return out

        # 2. Build prompt
        prompt = (
            f"IMAGE DESCRIPTION (verbatim):\n{data.image_cues}\n\n"
            "You are an investigative analyst interpreting the image for auditors.\n"
            "### OUTPUT FORMAT (strict)\n"
            "<one short reasoning paragraph>\n\n"
            "<≤2 sentences conveying the single most important takeaway>\n\n"
            "### RULES\n"
            "1. Quote any numbers/units exactly as given (e.g., \"42 kg\").\n"
            "2. Do NOT invent facts absent from the description.\n"
            "3. Keep each public sentence ≤20 words.\n"
            "4. You may think inside ##### blocks; that text will be removed.\n\n"
            "##### INTERNAL SCRATCHPAD (think here)\n#####\n\n"
            "Now generate the final answer:"
        )

        cot_and_msg = self._chat(prompt, temperature=0.3)

        # 3. Strip any echoed scratchpad
        cot_and_msg = re.sub(r"#####.*?#####", "", cot_and_msg, flags=re.DOTALL).strip()

        # 4. Separate CoT and public message
        parts = cot_and_msg.split("\n\n", 1)
        cot = parts[0]
        msg = parts[1] if len(parts) == 2 else cot  # fallback if no blank line

        # 5. Package result
        out = AgentOutput(cot=cot.strip(), result=msg.strip())
        logs[self.name] = out.__dict__ | {"raw": cot_and_msg}
        return out
