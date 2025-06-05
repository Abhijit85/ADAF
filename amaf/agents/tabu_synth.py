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

    # ── schema‑agnostic header/rows ------------------------------------------
    @staticmethod
    def _deduce_header_rows(tbl: Dict[str, Any]) -> tuple[List[str], List[List[str]]]:
        hdr, rows = tbl.get("header"), tbl.get("rows")
        if hdr is not None and rows is not None:
            return hdr, rows
        arr: List[List[str]] = tbl.get("table", [])
        arr = [r for r in arr if any(str(c).strip() for c in r)]
        return (arr[0], arr[1:]) if arr else ([], [])

    # ── markdown builder -----------------------------------------------------
    @staticmethod
    def _markdown(header: List[str], rows: List[List[str]]) -> str:
        if not header and not rows:
            return "(empty table)"
        if not header:
            header = ["" for _ in rows[0]]
        return (
            " | ".join(map(str, header)) + "\n"
            + " | ".join(["---"] * len(header)) + "\n"
            + "\n".join(" | ".join(map(str, r)) for r in rows)
        )

    # ── main run -------------------------------------------------------------
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:  # noqa: D401
        """Analyse the table and output structured forensic insights."""
        header, rows = self._deduce_header_rows(data.table)
        md_table = self._markdown(header, rows)

        prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
        prompt = prompt_template.format(md_table=md_table)

        # Temperature slightly above zero for richer insights but still stable.
        raw = self._chat(prompt, temperature=0.2).strip()

        # Remove scratchpad if model kept delimiters but echoed them.
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
