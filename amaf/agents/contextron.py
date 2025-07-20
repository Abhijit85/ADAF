# amaf/agents/contextron.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from ..core import AgentOutput, InputData
from .base import Agent


class ContextronAgent(Agent):
    """Summarise free‑form context into tagged insights."""

    TAG_SET = "[DEFINITION] [SCOPE] [SOURCE] [NOTE] [WARNING]"

    def __init__(self, dataset: str = None) -> None:
        super().__init__("Contextron", dataset)

    # ── main run ────────────────────────────────────────────────────────────
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # 1. No context? early‑return
        if not data.context:
            out = AgentOutput(cot="(no context)", result="")
            logs[self.name] = out.__dict__
            return out

        # 2. Build prompt from external template
        prompt_file = self.get_prompt_path("contextron.txt")
        prompt_template = prompt_file.read_text(encoding="utf-8")
        
        # Check if this is the new FinQA format with table_context and questions
        if "{table_context}" in prompt_template and "{questions}" in prompt_template:
            # Get table context from TabuSynth if available
            table_context = logs.get("TabuSynth", {}).get("result", "") or "(no table analysis available)"
            # Handle questions - use questions field if available, otherwise empty string
            questions = getattr(data, 'questions', ['']) or ['']
            if isinstance(questions, list):
                questions = questions[0] if questions else ''
            prompt = prompt_template.format(
                context=data.context, 
                table_context=table_context,
                questions=questions
            )
        else:
            # Use the original format
            prompt = prompt_template.format(context=data.context, tag_set=self.TAG_SET)
        
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
