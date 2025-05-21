import openai, os
from typing import Dict, Any
from .base import Agent, InputData

class TabuSynthAgent(Agent):
    """Extracts key facts from the table."""
    def __init__(self):
        super().__init__("TabuSynth")

    def run(self, data: InputData, logs: Dict[str, Any]) -> str:
        table = data.table
        headers = table.get("header", [])
        rows = table.get("rows", [])
        # Format table as markdown
        md = " | ".join(map(str, headers)) + "\n" + " | ".join(["---"]*len(headers)) + "\n"
        for r in rows:
            md += " | ".join(map(str, r)) + "\n"
        prompt = (
            f"You are a financial analyst. Here is a table:\n\n{md}\n\n"
            "List 3‑5 concise factual insights about this data."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4", temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        facts = response.choices[0].message.content.strip()
        logs[self.name] = {"facts": facts}
        return facts
