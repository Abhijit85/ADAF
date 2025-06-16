# amaf/agents/tabu_synth.py

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from ..core import AgentOutput, InputData
from .base import Agent

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "tabu_synth.txt"

class TabuSynthAgent(Agent):
    def __init__(self) -> None:
        super().__init__("TabuSynth")

    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        """Analyse the table and output structured forensic insights."""
        # Get table data directly from the table field
        table_data = data.table["table"]
        
        # Create a simple string representation of the table
        table_str = "\n".join(", ".join(map(str, row)) for row in table_data)

        prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
        prompt = prompt_template.format(table_str=table_str, questions=data.questions)

        # Temperature slightly above zero for richer insights but still stable
        raw = self._chat(prompt, temperature=0.2).strip()

        # Remove scratchpad if model kept delimiters but echoed them
        raw = re.sub(r"#####.*?#####", "", raw, flags=re.DOTALL)
        json_txt = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.DOTALL)

        try:
            obj = json.loads(json_txt)
            reasoning = obj.get("reasoning", "").strip()
            bullets = "\n".join(obj.get("bullets", [])).strip()
        except json.JSONDecodeError:
            logging.warning("TabuSynth: JSON parse failed; returning raw text")
            reasoning, bullets = "", json_txt.strip()

        out = AgentOutput(cot=reasoning, result=bullets)
        logs[self.name] = out.__dict__ | {"raw": raw}
        return out
