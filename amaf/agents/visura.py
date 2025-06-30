# amaf/agents/visura.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict
from PIL import Image
import pytesseract
import os
from glob import glob

from ..core import AgentOutput, InputData
from .base import Agent


class VisuraAgent(Agent):
    """Summarise `image_cues` into a two-sentence insight."""

    def __init__(self, dataset: str = None) -> None:
        super().__init__("Visura", dataset)

    def find_image_with_extension(self, base_path):
        # Try the path as given first
        if os.path.exists(base_path):
            return base_path
        # Try common image extensions
        for ext in [".JPG", ".jpg", ".jpeg", ".png"]:
            candidate = base_path + ext
            if os.path.exists(candidate):
                return candidate
        # Try glob for any extension
        matches = glob(base_path + ".*")
        if matches:
            return matches[0]
        return None

    # ── main run ────────────────────────────────────────────────────────
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        """Convert an image to text with OCR and summarise it, or use vision model if image is present."""

        # If image_path is present, send image to GPT-4o vision
        if data.image_path and len(data.image_path) > 0:
            image_path = self.find_image_with_extension(data.image_path[0])
            if image_path is not None:
                prompt_file = self.get_prompt_path("visura.txt")
                prompt_template = prompt_file.read_text(encoding="utf-8")
                prompt = prompt_template.format(image_cues="")  # or any context you want
                response = self._chat(prompt, temperature=0.3, image_path=image_path)
                out = AgentOutput(cot="", result=response)
                logs[self.name] = out.__dict__ | {"raw": response}
                return out
            else:
                print(f"ERROR: No image file found for base path {data.image_path[0]}")
                # Fallback to text-only logic below
        """
        # Fallback: use image_cues or OCR if available
        ocr_text = ""
        image_text = data.image_cues.strip()
        ""
        if not image_text and data.image_path:
            try:
                for image_path in data.image_path:
                    pass  # (old OCR logic can go here if needed)
            except Exception:
                image_text = ""
        if not image_text:
            out = AgentOutput(cot="(no visuals)", result="")
            logs[self.name] = out.__dict__
            return out
        """
        # 2. Build prompt from external template
        prompt_file = self.get_prompt_path("visura.txt")
        prompt_template = prompt_file.read_text(encoding="utf-8")
        prompt = prompt_template.format(image_cues=image_text)

        cot_and_msg = self._chat(prompt, temperature=0.3)

        # 3. Strip any echoed scratchpad
        cot_and_msg = re.sub(r"#####.*?#####", "", cot_and_msg, flags=re.DOTALL).strip()

        # 4. Separate CoT and public message
        parts = cot_and_msg.split("\n\n", 1)
        cot = parts[0]
        msg = parts[1] if len(parts) == 2 else cot  # fallback if no blank line

        # 5. Package result
        out = AgentOutput(cot=cot.strip(), result=msg.strip())
        logs[self.name] = out.__dict__ | {"raw": cot_and_msg, "ocr": ocr_text}
        return out
