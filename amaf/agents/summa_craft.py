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
        # 1. Collect all agent raw outputs separately
        tabu_synth_output = ""
        contextron_output = ""
        visura_output = ""
        
        # Get TabuSynth raw output
        tabu_synth_log = logs.get("TabuSynth", {})
        if tabu_synth_log:
            txt = tabu_synth_log.get("raw", "")  # Use raw instead of result
            txt = re.sub(r"^\[\w+] *", "", txt).strip()          # drop "[Agent]"
            tabu_synth_output = txt

        # Get Contextron raw output
        contextron_log = logs.get("Contextron", {})
        if contextron_log:
            txt = contextron_log.get("raw", "")  # Use raw instead of result
            txt = re.sub(r"^\[\w+] *", "", txt).strip()          # drop "[Agent]"
            contextron_output = txt

        # Get Visura raw output
        visura_log = logs.get("Visura", {})
        if visura_log:
            txt = visura_log.get("raw", "")  # Use raw instead of result
            txt = re.sub(r"^\[\w+] *", "", txt).strip()          # drop "[Agent]"
            visura_output = txt

        # Also gather individual bullet lines for Answer Echoes from processed results
        echo_lines = []
        for agent_name in ["TabuSynth", "Contextron", "Visura"]:
            agent_log = logs.get(agent_name, {})
            if agent_log:
                txt = agent_log.get("result", "")  # Still use result for echoes
                txt = re.sub(r"^\[\w+] *", "", txt).strip()
                if txt:
                    lines = [ln.strip() for ln in txt.splitlines() if ln.strip().lstrip().startswith("-")]
                    echo_lines.extend(lines)

        # 2. Audience-specific template
        templates = {
            "general": (
                "answer the questions precisely using the context from table and text agent "
                "Output the answer in bullet points."
                "also provide contextual insights."
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
        
        # Format questions for the prompt
        questions = getattr(data, "questions", None)
        if isinstance(questions, list):
            questions_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        elif isinstance(questions, str):
            questions_str = questions
        else:
            questions_str = ""

        prompt = prompt_template.format(
            tabu_synth_output=tabu_synth_output,
            contextron_output=contextron_output,
            visura_output=visura_output,
            questions=questions_str
        )

        # 4. Call LLM and post-process
        raw_resp = self._chat(prompt, temperature=0).strip()
        # strip scratchpad if the model echoed it
        cleaned_resp = re.sub(r"#####.*?#####", "", raw_resp, flags=re.DOTALL).strip()

        summary = canonicalise_numbers(cleaned_resp)

        # 5. Build Answer Echoes
        answer_echoes = "\n".join(echo_lines) or "(none)"

        final_text = f"{summary.rstrip()}\n\nAnswer Echoes:\n{answer_echoes}"

        # 6. Package result
        out = AgentOutput(cot="(omitted)", result=final_text)
        logs[self.name] = out.__dict__ | {"raw": raw_resp}
        return out
