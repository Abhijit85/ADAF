# amaf/agents/visura.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict
from PIL import Image
import pytesseract

from ..core import AgentOutput, InputData
from .base import Agent
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "visura.txt"


class VisuraAgent(Agent):
    """Summarise `image_cues` into a two-sentence insight."""

    def __init__(self) -> None:
        super().__init__("Visura")

    # ── main run ────────────────────────────────────────────────────────
    def run(self, data: InputData, logs: Dict[str, Any]) -> AgentOutput:
        """Convert an image to text with OCR and summarise it."""

        # 1. Determine visual text either from provided cues or via OCR
        ocr_text = ""
        image_text = data.image_cues.strip()

        if not image_text and data.image_path:
            try:
                img = Image.open(data.image_path)
                ocr_text = pytesseract.image_to_string(img)
                image_text = ocr_text.strip()
            except Exception:
                image_text = ""

        if not image_text:
            out = AgentOutput(cot="(no visuals)", result="")
            logs[self.name] = out.__dict__
            return out

        # 2. Build prompt from external template
        prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
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
