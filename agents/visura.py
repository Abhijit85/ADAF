import openai, os
from typing import Dict, Any
from .base import Agent, InputData

class VisuraAgent(Agent):
    """Interprets text descriptions of visuals to extract insights."""
    def __init__(self):
        super().__init__("Visura")

    def run(self, data: InputData, logs: Dict[str, Any]) -> str:
        if not data.image_cues:
            logs[self.name] = {"insights": ""}
            return ""
        prompt = (
            "A visual element is described below. Provide one key takeaway it conveys.\n\n"
            f"Description: {data.image_cues}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4", temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        insight = response.choices[0].message.content.strip()
        logs[self.name] = {"insights": insight}
        return insight
