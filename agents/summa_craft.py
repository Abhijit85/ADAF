import openai, os, textwrap
from typing import Dict, Any
from .base import Agent, InputData

class SummaCraftAgent(Agent):
    """Composes the final personalized summary."""
    def __init__(self):
        super().__init__("SummaCraft")

    def run(self, data: InputData, logs: Dict[str, Any]) -> str:
        facts = logs.get("TabuSynth", {}).get("facts", "")
        ctx = logs.get("Contextron", {}).get("insights", "")
        vis = logs.get("Visura", {}).get("insights", "")
        profile = data.user_profile or "general reader"
        content = textwrap.dedent(f"""        USER PROFILE: {profile}
        FACTS:
        {facts}

        CONTEXT INSIGHTS:
        {ctx}

        VISUAL INSIGHTS:
        {vis}
        """)
        messages = [
            {"role": "system", "content": "You are SummaCraft, an expert summarizer."},
            {"role": "user", "content": content + "\nWrite a clear, concise summary tailored to the user profile."}
        ]
        response = openai.ChatCompletion.create(model="gpt-4", temperature=0, messages=messages)
        summary = response.choices[0].message.content.strip()
        logs[self.name] = {"summary": summary}
        return summary
