# amaf/agents/summa_craft.py 

from __future__ import annotations

import re
from pathlib import Path
import textwrap
from typing import Any, Dict

from ..core import AgentOutput, InputData
from .base import Agent


# ───────── helpers ─────────────────────────────────────────────────────
def canonicalise_numbers(text: str) -> str:
    """Remove thin spaces / NBSP and digit-group commas (60,000 → 60000)."""
    text = re.sub(r"[\u202F\u00A0]", "", text)              # thin / nbsp
    return re.sub(r"(?<=\d),(?=\d)", "", text)              # commas


# ───────── main agent ─────────────────────────────────────────────────
class SummaCraftAgent(Agent):
    """Synthesize upstream notes into a concise, audience-aware summary."""

    def __init__(self, dataset: str = None) -> None:
        super().__init__("SummaCraft", dataset)

    # ------------------------------------------------------------------
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        # 1. Collect upstream notes (TabuSynth, Contextron, …)
        raw_blocks: list[str] = []
        for key in ("TabuSynth", "Contextron", "Visura",
                    "TrendAnalyser", "TopKFilter"):
            txt = logs.get(key, {}).get("result", "")
            txt = re.sub(r"^\[\w+] *", "", txt).strip()          # drop "[Agent]"
            if txt:
                raw_blocks.append(txt)

        bundle_for_reasoning = "\n\n".join(raw_blocks) or "(no notes)"

        # Also gather individual bullet lines for Answer Echoes
        echo_lines = [
            ln.strip() for blk in raw_blocks
            for ln in blk.splitlines()
            if ln.strip().lstrip().startswith("-")
        ]

        # Early path: tatqa deterministic summary --------------------------------
        if self.dataset == "tatqa" and logs.get("Calculator", {}).get("json"):
            merged = logs["Calculator"]["json"]
            answers = []
            echoes_lines = []
            for idx, item in enumerate(merged, 1):
                ans = item.get("text", "").strip()
                if idx-1 < len(data.answer_types):
                    if data.answer_types[idx-1] == "percent" and not ans.endswith("%"):
                        ans += "%"
                if idx-1 < len(data.answer_scales):
                    sc = data.answer_scales[idx-1]
                    if sc and sc not in ans:
                        ans += f" {sc}"
                if not ans:
                    ans = "INSUFFICIENT DATA"
                answers.append(f"{idx}) {ans}")

                tbl_reason = logs.get("TabuSynth", {}).get("json", [])
                tbl_line = tbl_reason[idx-1]["reasoning"] if idx-1 < len(tbl_reason) else "(none)"
                ctx_line = "(none)"
                tag = item.get("status", "")
                calc = item.get("calc_answer", "")
                calc_part = f" | CALC: {calc}" if calc else ""
                echoes_lines.append(f"{idx}) TABLE: {tbl_line} | CONTEXT: {ctx_line}{calc_part} | TAG: {tag}")

            final_text = "\n".join(answers) + "\n\nAnswer Echoes:\n" + "\n".join(echoes_lines)
            logs.setdefault(self.name, {})["prompt"] = "(deterministic)"
            logs[self.name]["input"] = f"merged_items={len(merged)}"
            out = AgentOutput(cot="", result=final_text)
            logs[self.name]["result"] = final_text
            return out

        # 2. Audience-specific template
        templates = {
            "general": (
                 "Answer the questions in answer echoes."
                 "There should be as many answer echoes as there are "
                "questions."
            ),
            "retail investor": (
                "Explain in plain language why these figures matter to a non-expert "
                "investor. Use ≤180 words, same structure."
            ),
            "analyst": (
                "Provide an expert synthesis focusing on material trends, anomalies, "
                "and valuation drivers. ≤200 words."
            ),
        }
        template = templates.get(data.user_profile, templates["general"])
        
        # 3. Build prompt from external template
        prompt_file = self.get_prompt_path("summa_craft.txt")
        prompt_template = prompt_file.read_text(encoding="utf-8")
        
        # Detect whether the template expects a {questions} placeholder (e.g., TATQA)
        if "{questions}" in prompt_template:
            q_list = getattr(data, "questions", None)
            if q_list:
                questions = "\n".join(q_list)
                prompt = prompt_template.format(
                    template=template,
                    bundle_for_reasoning=bundle_for_reasoning,
                    questions=questions,
                )
            else:
                print(f"[WARN] No questions found for input UID={data.table.get('uid','?')}")
                prompt = prompt_template.format(
                    template=template,
                    bundle_for_reasoning=bundle_for_reasoning,
                )
        else:
            prompt = prompt_template.format(
                template=template,
                bundle_for_reasoning=bundle_for_reasoning,
            )

        # 4. Call LLM and post-process
        logs.setdefault(self.name, {})["prompt"] = prompt
        raw_resp = self._chat(prompt, temperature=0).strip()
        # strip scratchpad if the model echoed it
        cleaned_resp = re.sub(r"#####.*?#####", "", raw_resp, flags=re.DOTALL).strip()

        summary = canonicalise_numbers(cleaned_resp)

        # 5. Build Answer Echoes
        answer_echoes = "\n".join(echo_lines) or "(none)"

        # ---- Custom output path for TATQA multihop -----------------------
        if self.dataset == "tatqa" and logs.get("Calculator", {}).get("json"):
            merged = logs["Calculator"]["json"]
            answers: list[str] = []
            echoes_lines: list[str] = []

            for idx, item in enumerate(merged, 1):
                ans = item.get("text", "").strip()
                # Enforce correct scale / type formatting
                if idx-1 < len(data.answer_types):
                    ans_type = data.answer_types[idx-1]
                    if ans_type == "percent" and not ans.endswith("%"):
                        ans += "%"
                if idx-1 < len(data.answer_scales):
                    scale_kw = data.answer_scales[idx-1]
                    if scale_kw and scale_kw not in ans:
                        ans = f"{ans} {scale_kw}" if ans else ans
                if not ans:
                    ans = "INSUFFICIENT DATA"
                answers.append(f"{idx}) {ans}")

                # Build echoes
                tbl_line = "(none)"
                ctx_line = "(none)"
                if idx-1 < len(raw_blocks):
                    # crude pick: first bullet line containing a number
                    for ln in raw_blocks[idx-1].splitlines():
                        if any(ch.isdigit() for ch in ln):
                            if ln.startswith("-"):
                                ctx_line = ln[1:].strip()
                            break
                # Table bullet from TabuSynth results if available
                tbl_json = logs.get("TabuSynth", {}).get("json", [])
                if idx-1 < len(tbl_json):
                    tbl_line = tbl_json[idx-1].get("reasoning", "")[:80] or "(none)"

                tag_line = item.get("status", "")
                calc_ans = item.get("calc_answer", "")
                calc_part = f" | CALC: {calc_ans}" if calc_ans else ""
                echoes_lines.append(
                    f"{idx}) TABLE: {tbl_line} | CONTEXT: {ctx_line}{calc_part} | TAG: {tag_line}"
                )

            final_text = "\n".join(answers) + "\n\nAnswer Echoes:\n" + "\n".join(echoes_lines)
        else:
            final_text = f"{summary.rstrip()}\n\nAnswer Echoes:\n{answer_echoes}"

        # 6. Package result
        out = AgentOutput(cot="(omitted)", result=final_text)
        existing = logs.get(self.name, {})
        logs[self.name] = existing | out.__dict__ | {"raw": raw_resp}
        return out
