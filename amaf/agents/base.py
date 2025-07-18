from abc import ABC, abstractmethod
from typing import Dict, Any
from ..core import InputData, AgentOutput
import openai
import os
import mimetypes
import base64
from pathlib import Path


class Agent(ABC):
    name: str

    def __init__(self, name: str, dataset: str = None):
        self.name = name
        self.dataset = dataset
        # override default model via env or fallback to valid default
        default_model = "gpt-4o"
        self.model = os.getenv("AMAF_MODEL", default_model)
    # ensure the OPENAI_API_KEY is set in the environment

    def get_prompt_path(self, prompt_filename: str) -> Path:
        """Get the path to a prompt file, preferring dataset-specific version."""
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        
        # Try dataset-specific path first
        if self.dataset:
            dataset_prompt_path = prompts_dir / self.dataset / prompt_filename
            if dataset_prompt_path.exists():
                return dataset_prompt_path
        
        # Fallback to default prompt
        return prompts_dir / prompt_filename

    def _chat(self, prompt: str, temperature: float = .0, image_path: str = None) -> str:
        if image_path:
            mime_type, _ = mimetypes.guess_type(image_path)
            with open(image_path, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode()
            resp = openai.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}}
                    ]
                }],
                temperature=temperature,
            )
        else:
            resp = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        return resp.choices[0].message.content.strip()

    @abstractmethod
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        pass
    