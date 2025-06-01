from pathlib import Path, PurePath
import yaml, jinja2
from .base import Agent, InputData

class MCPController(Agent):
    """Orchestrator driven by a YAML Model‑Control Protocol."""
    def __init__(self, proto_file: str, registry: dict[str, Agent]):
        super().__init__("MCP")
        self.proto = yaml.safe_load(Path(proto_file).read_text())
        self.registry = registry
        self.env = jinja2.Environment()

    def _cond_ok(self, expr: str, data: InputData):
        if not expr:
            return True
        tmpl = self.env.from_string("{{" + expr + "}}")
        return tmpl.render(input=data.__dict__) == "True"

    def run(self, data: InputData, logs: dict):
        summary = ""
        for step in self.proto["steps"]:
            agent_name = step["agent"]
            cond = step.get("when", "")
            if self._cond_ok(cond, data):
                summary = self.registry[agent_name].run(data, logs)
        return summary