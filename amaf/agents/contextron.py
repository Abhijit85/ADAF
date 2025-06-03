from typing import Dict, Any
from ..core import AgentOutput, InputData
from .base import Agent


class ContextronAgent(Agent):
    def __init__(self):
        super().__init__("Contextron")

    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # ── 1. No context? early-return ───────────────────────────────────
        if not data.context:
            out = AgentOutput(cot="(no context)", result="")
            logs[self.name] = out.__dict__
            return out

        # ── 2. Build the prompt ──────────────────────────────────────────
        prompt = (
            f"Context:\n{data.context}\n\n"
            "Summarise only the information that clarifies the table above.\n"
            "First write a short reasoning paragraph, leave a blank line, "
            "then list the insights as bullet points."
        )

        cot_and_ins = self._chat(prompt)  # default temperature = 0

        # ── 3. Robustly separate CoT and insights ────────────────────────
        parts = cot_and_ins.split("\n\n", 1)
        if len(parts) == 2:
            cot, insight_block = parts
        else:                              # model returned single block
            cot, insight_block = "", cot_and_ins

        out = AgentOutput(
            cot=cot.strip(),
            result=insight_block.strip()
        )

        # ── 4. Record structured + raw in the shared log ────────────────
        logs[self.name] = out.__dict__
        logs[self.name]["raw"] = cot_and_ins

        return out
