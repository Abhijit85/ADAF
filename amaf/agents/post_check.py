from __future__ import annotations

import re
from typing import Any, Dict

from ..core import AgentOutput, InputData
from .base import Agent


class PostCheckAgent(Agent):
    """Deterministic audit pass – no LLM calls."""

    def __init__(self, dataset: str = None) -> None:
        super().__init__("PostCheck", dataset)

    # ------------------------------------------------------------------
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # Grab artefacts
        summa_txt: str = logs.get("SummaCraft", {}).get("result", "")
        merged = logs.get("Calculator", {}).get("json", [])
        entities_lists = [e.get("entities", []) for e in logs.get("TabuSynth", {}).get("json", [])]

        issues: list[str] = []

        # Split answer lines (prefix "1) ")
        answer_lines = [ln for ln in summa_txt.splitlines() if re.match(r"\d+\)", ln)]

        for idx, line in enumerate(answer_lines):
            q_idx = idx  # zero-based
            # --- Entity coverage ---
            ents = []
            if q_idx < len(entities_lists):
                ents = entities_lists[q_idx]
            missing = [e for e in ents if e and e.lower() not in line.lower()]
            if missing:
                issues.append(f"{idx+1}) [MISSING_ENTITY] {', '.join(missing)}")

            # --- Scale/type conformity ---
            if q_idx < len(data.answer_scales):
                scale = data.answer_scales[q_idx]
                if scale and scale not in line:
                    issues.append(f"{idx+1}) [SCALE_MISMATCH] expected '{scale}'")
            if q_idx < len(data.answer_types):
                atype = data.answer_types[q_idx]
                if atype == "percent" and "%" not in line:
                    issues.append(f"{idx+1}) [TYPE_MISMATCH] expected % symbol")

            # --- Completion status ---
            status = merged[q_idx].get("status", "") if q_idx < len(merged) else ""
            if status != "FULL":
                issues.append(f"{idx+1}) [INSUFFICIENT_DATA] status={status}")

        summary = "POST-CHECK SUMMARY (counts): " + \
            f"missing_entity={sum('MISSING_ENTITY' in i for i in issues)}, " \
            f"scale/type_mismatch={sum('MISMATCH' in i for i in issues)}, " \
            f"insufficient={sum('INSUFFICIENT_DATA' in i for i in issues)}"

        # Append issues to logs
        logs[self.name] = {
            "issues": issues,
            "summary": summary,
        }
        out_text = summary + "\n" + "\n".join(issues)
        return AgentOutput(cot="", result="") 