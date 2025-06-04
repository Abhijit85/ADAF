# amaf/agents/tabu_synth.py  – robust to TAT-QA schema

import json
import re
import logging
from typing import Dict, Any, List

from ..core import AgentOutput, InputData
from .base import Agent


class TabuSynthAgent(Agent):
    def __init__(self):
        super().__init__("TabuSynth")

    # ── schema-agnostic header/rows ------------------------------------
    def _deduce_header_rows(self, tbl: Dict[str, Any]) -> tuple[list, list]:
        hdr, rows = tbl.get("header"), tbl.get("rows")
        if hdr is not None and rows is not None:
            return hdr, rows
        arr: List[List[str]] = tbl.get("table", [])
        arr = [r for r in arr if any(str(c).strip() for c in r)]
        return (arr[0], arr[1:]) if arr else ([], [])

    # ── markdown builder ----------------------------------------------
    def _markdown(self, header, rows) -> str:
        if not header and not rows:
            return "(empty table)"
        if not header:
            header = ["" for _ in rows[0]]
        return (
            " | ".join(map(str, header)) + "\n"
            " | ".join(["---"] * len(header)) + "\n"
            + "\n".join(" | ".join(map(str, r)) for r in rows)
        )

    # ── main run -------------------------------------------------------
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        header, rows = self._deduce_header_rows(data.table)
        md_table = self._markdown(header, rows)

        prompt = (
            "You are a forensic accountant auditing table.\n"
            "RULES\n"
            "• Use ONLY numbers that appear verbatim in the table.\n"
            "• No invented columns (GDP, population, etc.).\n\n"
            f"TABLE (Markdown):\n{md_table}\n\n"
            "TASK\n"
            "1. Think step-by-step about relevant quantitative facts.\n"
            "2. Select **6-10** insights; each bullet must quote the exact "
            "numbers and units.\n"
            "3. Return VALID JSON ONLY with keys `reasoning` and `bullets`.\n"
            "   Example:\n"
            '{ "reasoning": "...", "bullets": ["- cash ↓0.2 (16.0→15.8)", "..."] }'
        )

        raw = self._chat(prompt, temperature=0).strip()
        json_txt = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.DOTALL)

        try:
            obj = json.loads(json_txt)
            reasoning = obj.get("reasoning", "").strip()
            bullets = "\n".join(obj.get("bullets", [])).strip()
        except json.JSONDecodeError:
            logging.warning("TabuSynth: JSON parse failed")
            reasoning, bullets = "", json_txt.strip()

        out = AgentOutput(cot=reasoning, result=bullets)
        logs[self.name] = out.__dict__ | {"raw": raw}
        return out
