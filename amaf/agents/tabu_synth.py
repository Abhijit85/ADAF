# amaf/agents/tabu_synth.py  – robust to TAT-QA schema
import json
import re
import logging
from typing import Dict, Any, List

from ..core import AgentOutput, InputData
from .base import Agent


class TabuSynthAgent(Agent):
    def __init__(self) -> None:
        super().__init__("TabuSynth")

    # ------------------------------------------------------------------ #
    def _deduce_header_rows(self, tbl: Dict[str, Any]) -> tuple[list, list]:
        """Return (header, rows) regardless of schema."""
        # Case 1 – native {header:…, rows:…}
        hdr = tbl.get("header")
        rows = tbl.get("rows")
        if hdr is not None and rows is not None:
            return hdr, rows

        # Case 2 – TAT-QA / FinTabNet style: { "table": [[…], …] }
        arr: List[List[str]] = tbl.get("table", [])
        arr = [r for r in arr if any(str(cell).strip() for cell in r)]  # drop blank rows
        if not arr:
            return [], []

        return arr[0], arr[1:]  # first row = header, rest = body

    # ------------------------------------------------------------------ #
    def _markdown(self, header, rows) -> str:
        if not header and not rows:
            return "(empty table)"
        if not header:
            header = ["" for _ in rows[0]]
        md = (
            " | ".join(map(str, header))         + "\n" +
            " | ".join(["---"] * len(header))    + "\n" +
            "\n".join(" | ".join(map(str, r)) for r in rows)
        )
        return md

    # ------------------------------------------------------------------ #
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        header, rows = self._deduce_header_rows(data.table)
        md_table = self._markdown(header, rows)

        prompt = (
            "You are a forensic accountant auditing table.\n"
            "Use ONLY numbers that appear verbatim in the table.\n\n"
            f"TABLE (Markdown):\n{md_table}\n\n"
            "TASK\n"
            "1. Think step-by-step about which quantitative facts matter most "
            "(changes, magnitudes, unusual ratios).\n"
            "2. Select EXACTLY 3-6 insights; each bullet must quote the cell "
            "values it is based on.\n\n"
            "OUTPUT\n"
            "Return valid JSON with two keys:\n"
            '{ "reasoning": "<thoughts>", "bullets": ["- …", "- …"] }\n'
            "Only output JSON—no additional text."
        )

        raw = self._chat(prompt, temperature=0).strip()
        clean = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.DOTALL)

        try:
            j = json.loads(clean)
            bullets = "\n".join(j.get("bullets", []))
            cot = j.get("reasoning", "")
        except json.JSONDecodeError:
            logging.warning("TabuSynth: JSON parse failed, fallback to raw text")
            cot, bullets = "", clean

        out = AgentOutput(cot=cot.strip(), result=bullets.strip())
        logs[self.name] = out.__dict__
        logs[self.name]["raw"] = raw
        return out
