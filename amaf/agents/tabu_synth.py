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
    def __init__(self, dataset: str = None) -> None:
        super().__init__("TabuSynth", dataset)

    # ── schema‑agnostic header/rows ------------------------------------------
    @staticmethod
    def _deduce_header_rows(tbl: Any) -> tuple[List[str], List[List[str]]]:
        if isinstance(tbl, dict):
            hdr, rows = tbl.get("header"), tbl.get("rows")
            if hdr is not None and rows is not None:
                return hdr, rows
            arr: List[List[str]] = tbl.get("table", [])
        elif isinstance(tbl, list):
            arr = tbl  # type: ignore[assignment]
        else:
            return ([], [])

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

        prompt_file = self.get_prompt_path("tabu_synth.txt")
        prompt_template = prompt_file.read_text(encoding="utf-8")

        if self.dataset == "tatqa" and "{ctx_status}" in prompt_template:
            # ---- Build TATQA-specific prompt ---------------------------------
            table_str = "\n".join([" | ".join(map(str, row)) for row in [header] + rows if row])
            q_list = getattr(data, "questions", None)
            if q_list:
                questions = "\n".join(q_list)
            else:
                questions = getattr(data, "question", "")

            ctx_status_json = json.dumps(logs.get("Contextron", {}).get("json", []), ensure_ascii=False)

            prompt = prompt_template.format(table_str=table_str, questions=questions, ctx_status=ctx_status_json)
            logs.setdefault(self.name, {})["prompt"] = prompt
            logs[self.name]["input"] = table_str[:100] + ("…" if len(table_str) > 100 else "")

            raw = self._chat(prompt, temperature=0.2).strip()
            raw_clean = re.sub(r"#####.*?#####", "", raw, flags=re.DOTALL).strip()

            try:
                parsed = json.loads(raw_clean)
            except Exception:
                parsed = []

            out = AgentOutput(cot="", result="(json emitted)")
            existing = logs.get(self.name, {})
            logs[self.name] = existing | out.__dict__ | {"raw": raw, "json": parsed}
            return out

        # ---- Original path for finqa/mmqa/other -----------------------------
        if "{table_str}" in prompt_template and "{questions}" in prompt_template:
            table_str = "\n".join([" | ".join(map(str, row)) for row in [header] + rows if row])
            questions = getattr(data, "question", "") or ""
            prompt = prompt_template.format(table_str=table_str, questions=questions)
            logs.setdefault(self.name, {})["prompt"] = prompt
            logs[self.name]["input"] = md_table[:100] + ("…" if len(md_table) > 100 else "")
        else:
            prompt = prompt_template.format(md_table=md_table)

        raw = self._chat(prompt, temperature=0.2).strip()

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
        existing = logs.get(self.name, {})
        logs[self.name] = existing | out.__dict__ | {"raw": raw}
        return out
