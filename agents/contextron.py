import openai, os
from typing import Dict, Any
from .base import Agent, InputData

class ContextronAgent(Agent):
    """Extracts relevant insights from surrounding text."""
    def __init__(self):
        super().__init__("Contextron")

    def run(self, data: InputData, logs: Dict[str, Any]) -> str:
        if not data.context:
            logs[self.name] = {"insights": ""}
            return ""
        prompt = (
            "You are summarizing supporting report text for a financial table. " 
            "Focus only on points directly explaining or expanding the table data.\n\n"
            f"Context:\n{data.context}\n\nGive a short bullet list of the main insights."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4", temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        insights = response.choices[0].message.content.strip()
        logs[self.name] = {"insights": insights}
        return insights
