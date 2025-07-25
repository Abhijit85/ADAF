# amaf/agents/tabu_synth.py

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from ..core import AgentOutput, InputData
from .base import Agent

class TabuSynthAgent(Agent):
    def __init__(self, dataset: str = None, provider="openai", model="gpt-3.5-turbo", method=None):
        super().__init__("TabuSynth", dataset, provider=provider, model=model, method=method)

    
    # ── main run -------------------------------------------------------------
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:  # noqa: D401
        """Analyse the table and output structured forensic insights."""

        prompt_file = self.get_prompt_path("tabu_synth.txt")
        prompt_template = prompt_file.read_text(encoding="utf-8")
        
        # Check if this is the new FinQA format with table_str and questions
        if "{table_str}" in prompt_template and "{questions}" in prompt_template:
            # Convert table to simple string format for FinQA
            table_str = str(data.table)
            # Handle questions - support both list and dict formats
            questions = getattr(data, 'questions', ['']) or ['']
            if isinstance(questions, dict):
                # For TATQA format with qid -> question mapping
                questions_text = '\n'.join([f"QID: {qid} - {question}" for qid, question in questions.items()])
            elif isinstance(questions, list):
                questions_text = questions[0] if questions else ''
            else:
                questions_text = str(questions)
            prompt = prompt_template.format(table_str=table_str, questions=questions_text)
        else:
            # Use the original format
            prompt = prompt_template.format(md_table=data.table)

        # Temperature slightly above zero for richer insights but still stable.
        raw = self._chat(prompt, temperature=0.2).strip()

        # Remove scratchpad if model kept delimiters but echoed them.
        raw = re.sub(r"#####.*?#####", "", raw, flags=re.DOTALL)
        
        # For FinQA format, don't try to parse JSON - use raw output
        if "{table_str}" in prompt_template and "{questions}" in prompt_template:
            reasoning = raw.strip()
            bullets = raw.strip()
        else:
            # Original JSON parsing for other formats
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
