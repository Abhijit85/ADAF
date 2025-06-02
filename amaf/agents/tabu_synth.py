from ..core import InputData, AgentOutput
from .base import Agent

class TabuSynthAgent(Agent):
    def __init__(self):
        super().__init__("TabuSynth")

    def run(self, data, logs):
        # ➜ format table to Markdown for GPT clarity
        hdr, rows = data.table.get("header", []), data.table.get("rows", [])
        md = " | ".join(map(str, hdr))+"\n"+" | ".join(["---"]*len(hdr))+"\n"+"\n".join(
            " | ".join(map(str, r)) for r in rows)
        prompt = f"""Analyse the table below and list 3‑6 key quantitative insights.\n
TABLE:\n{md}\n
Think step‑by‑step, output as bullet points."""
        cot_and_res = self._chat(prompt)
        logs[self.name] = {"raw": cot_and_res}
        # simple split: first paragraph = CoT, last bullet list = facts
        cot, *facts = cot_and_res.split("\n\n")
        return AgentOutput(cot=cot.strip(), result="\n".join(facts).strip())