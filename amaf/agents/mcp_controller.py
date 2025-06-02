from pathlib import Path
import yaml
import jinja2
from typing import Dict
from .base import Agent
from ..core import InputData, AgentOutput


class MCPController(Agent):
    """YAML/Jinja2-driven orchestrator."""
    def __init__(self, proto_file: str, registry: Dict[str, Agent]):
        super().__init__("MCP")
        self.proto = yaml.safe_load(Path(proto_file).read_text())
        self.registry = registry
        self.env = jinja2.Environment()

    # ---------- helpers ----------
    def _cond_ok(self, expr: str, data: InputData) -> bool:
        """Evaluate `when:` Jinja expression against InputData."""
        if not expr:                       # empty string → unconditional
            return True
        rendered = self.env.from_string(expr).render(input=data.__dict__)
        # Accept truthy / falsy   ("True" / "False" / "1" / "")
        try:
            rendered_val = yaml.safe_load(rendered)  # cast "False" → False
        except yaml.YAMLError:
            rendered_val = rendered
        return bool(rendered_val)

    # ---------- main ----------
    def run(self, data: InputData, logs: Dict) -> str:
        summary = ""
        for step in self.proto["steps"]:
            agent_name = step["agent"]
            if self._cond_ok(step.get("when", ""), data):
                out: AgentOutput = self.registry[agent_name].run(data, logs)
                # keep latest summary if the agent supplies one
                if out.result:
                    summary = out.result
        return summary
    
