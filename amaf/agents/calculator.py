# amaf/agents/contextron.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict
from decimal import Decimal, ROUND_HALF_UP
import ast
import operator
import json

from ..core import AgentOutput, InputData
from .base import Agent


class CalculatorAgent(Agent):
    """Perform calculations."""

    TAG_SET = "[DEFINITION] [SCOPE] [SOURCE] [NOTE] [WARNING]"

    def __init__(self, dataset: str = None) -> None:
        super().__init__("Calculator", dataset)

    # ── main run ────────────────────────────────────────────────────────────
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # -------- Fast path for TAT-QA multihop ---------------------------
        if self.dataset == "tatqa":
            tbl_json = logs.get("TabuSynth", {}).get("json", [])
            ctx_json = logs.get("Contextron", {}).get("json", [])

            if not tbl_json and not ctx_json:
                # fallback to original prompt-driven path below
                pass
            else:
                merged: list[Dict[str, Any]] = []
                q_count = len(getattr(data, "questions", []))
                for i in range(q_count):
                    entry = {}
                    if i < len(ctx_json):
                        entry.update(ctx_json[i])
                    if i < len(tbl_json):
                        # table answers override
                        entry.update(tbl_json[i])
                    merged.append(entry)

                # Evaluate expressions
                for idx, ent in enumerate(merged):
                    expr = (ent.get("expr") or "").strip()
                    if expr and "[CALC]" in expr:
                        expr_clean = expr.replace("[CALC]", "").strip()
                        try:
                            val = _safe_eval(expr_clean)
                            # Preserve symbols present in expression, but do not append scale words.
                            out_str = str(val.normalize())
                            if "%" in expr_clean and not out_str.endswith("%"):
                                out_str += "%"
                            if "$" in expr_clean and not out_str.startswith("$"):
                                out_str = "$ " + out_str
                            ent["calc_answer"] = out_str
                            ent["status"] = "FULL"
                            ent["text"] = out_str
                            ent["expr"] = ""
                        except Exception:
                            ent["calc_answer"] = "ERROR"

                # Store
                out = AgentOutput(cot="", result="(calc done)")
                existing = logs.get(self.name, {})
                logs[self.name] = existing | out.__dict__ | {"json": merged}
                return out

        # 1. Build prompt for LLM (fallback path) ----------------------------
        prompt_file = self.get_prompt_path("calculator.txt")
        prompt_template = prompt_file.read_text(encoding="utf-8")

        if self.dataset == "tatqa" and "{items_json}" in prompt_template:
            items_json = json.dumps(logs.get("TabuSynth", {}).get("json", []), ensure_ascii=False)
            prompt = prompt_template.format(items_json=items_json)
        elif self.dataset == "tatqa" and "{table_context}" in prompt_template:
            # --- Use table + context bullets from upstream agents ---------
            table_context = logs.get("TabuSynth", {}).get("result", "") or "(no table analysis)"
            text_context = logs.get("Contextron", {}).get("result", "") or "(no text analysis)"

            if not table_context and logs.get("TabuSynth", {}).get("json"):
                # Fallback: synthesise simple bullets from JSON expr
                table_context = str(logs["TabuSynth"]["json"])

            questions_block = "\n".join(data.questions) if data.questions else (data.question or "")

            prompt = prompt_template.format(
                table_context=table_context,
                text_context=text_context,
                questions=questions_block,
            )
        else:
            # Generic datasets keep original placeholder contract
            prompt = prompt_template.format(context=data.context, tag_set=self.TAG_SET)

        # Log prompt for debugging
        logs.setdefault(self.name, {})["prompt"] = prompt[:100] + ("…" if len(prompt) > 100 else "")
        logs[self.name]["input"] = "items_len=" + str(len(logs.get("TabuSynth", {}).get("json", [])))
        cot_and_ins = self._chat(prompt, temperature=0)

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
        existing = logs.get(self.name, {})
        logs[self.name] = existing | out.__dict__ | {"raw": cot_and_ins}
        return out

# ---------------------------------------------------------------------------
# Safe expression evaluator --------------------------------------------------
# ---------------------------------------------------------------------------


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _safe_eval(expr: str) -> Decimal:
    """Evaluate arithmetic expression safely, returning Decimal."""

    # Remove currency symbols and commas
    clean = re.sub(r"[,$]", "", expr)

    node = ast.parse(clean, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Num):  # type: ignore[attr-defined]
            return Decimal(str(n.n))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError("Unsupported expression")

    result = _eval(node)
    return result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# (Formatting helper removed – scale/type enforcement happens in SummaCraft)
