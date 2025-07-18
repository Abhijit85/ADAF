# amaf/agents/contextron.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict
import json

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
        
        # Check if this is the new TATQA format with table_context and questions
        if "{table_context}" in prompt_template and "{questions}" in prompt_template:
            # Get table context from TabuSynth if available
            table_context = logs.get("TabuSynth", {}).get("result", "") or "(no table analysis available)"
            questions = getattr(data, 'question', '') or ''
            prompt = prompt_template.format(
                context=data.context, 
                table_context=table_context,
                questions=questions
            )
        else:
            # Use the original format
            prompt = prompt_template.format(context=data.context, tag_set=self.TAG_SET)
        
        logs.setdefault(self.name, {})["prompt"] = prompt
        cot_and_ins = self._chat(prompt, temperature=0.25)

        # 3. Remove any internal scratchpad echoed back
        cleaned = re.sub(r"#####.*?#####", "", cot_and_ins, flags=re.DOTALL).strip()
        # Store input preview (context first 120 chars)
        logs[self.name]["input"] = data.context[:100] + ("…" if len(data.context) > 100 else "")

        # Validate presence of questions for tatqa early – missing questions halt pipeline
        if self.dataset == "tatqa" and not getattr(data, "questions", None):
            raise ValueError("[Contextron] No questions provided for TAT-QA input; aborting pipeline.")

        # ---- Dataset-specific handling ------------------------------------
        if self.dataset == "tatqa":
            # Expect one JSON object per line (v2 spec).
            json_objs = []
            for line in cleaned.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    json_objs.append(json.loads(line))
                except Exception:
                    # not valid JSON line → ignore / fallback
                    pass

            if json_objs:
                out = AgentOutput(cot="", result="(json emitted)")
                existing = logs.get(self.name, {})
                logs[self.name] = existing | out.__dict__ | {
                    "prompt": prompt[:100] + ("…" if len(prompt) > 100 else ""),
                    "input": (data.context or "")[:100] + ("…" if len(data.context) > 100 else ""),
                    "raw": cot_and_ins,
                    "json": json_objs,
                }
                return out

        # ---- Fallback to original behaviour ------------------------------
        parts = cleaned.split("\n\n", 1)
        cot, insight_block = parts if len(parts) == 2 else ("", cleaned)

        out = AgentOutput(cot=cot.strip(), result=insight_block.strip())
        existing = logs.get(self.name, {})
        logs[self.name] = existing | out.__dict__ | {"raw": cot_and_ins}
        return out
