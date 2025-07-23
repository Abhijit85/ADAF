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

    def __init__(self, name: str, dataset: str = None, provider="openai", model="gpt-3.5-turbo", method: str = None):
        self.name = name
        self.dataset = dataset
        self.provider = provider
        self.model = model
        self.method = method

        print(f"DEBUG: Initializing agent {self.name} with provider={self.provider}, model={self.model}")

        if provider == "mistral":
            from mistralai import Mistral
            self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        elif provider == "gemini":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        elif provider == "lambda":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["LAMBDA_API_KEY"],
                                 base_url="https://api.lambda.ai/v1")
        else:  # openai
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        

    def get_prompt_path(self, prompt_filename: str) -> Path:
        """Get the path to a prompt file, preferring method- and dataset-specific version."""
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        
        # Try method + dataset-specific path first
        if self.dataset and self.method:
            method_prompt_path = prompts_dir / self.dataset / self.method / prompt_filename
            if method_prompt_path.exists():
                return method_prompt_path

        # Try dataset-specific path
        if self.dataset:
            dataset_prompt_path = prompts_dir / self.dataset / prompt_filename
            if dataset_prompt_path.exists():
                return dataset_prompt_path

        # Fallback to default prompt
        return prompts_dir / prompt_filename

    def _chat(self, prompt: str, temperature: float = .0, image_path: str = None) -> str:
        print(f"DEBUG: _chat called with provider={self.provider}, model={self.model}")
        
        if self.provider == "mistral":
            print(f"DEBUG: Using Mistral client.chat.complete() with model={self.model}")
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            print(f"DEBUG: Mistral API call successful, response type: {type(response)}")
            return response.choices[0].message.content
        else:
            print(f"DEBUG: Using OpenAI client.chat.completions.create() with model={self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                #temperature=temperature,
            )
            print(f"DEBUG: OpenAI API call successful, response type: {type(response)}")
            return response.choices[0].message.content

    @abstractmethod
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        pass
    