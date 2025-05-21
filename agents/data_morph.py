from typing import Dict, Any
from .base import Agent, InputData
from .tabu_synth import TabuSynthAgent
from .contextron import ContextronAgent
from .visura import VisuraAgent
from .summa_craft import SummaCraftAgent

class DataMorphAgent(Agent):
    """Orchestrates the full AMAF pipeline adaptively."""
    def __init__(self,
                 tabu: TabuSynthAgent,
                 ctx: ContextronAgent,
                 vis: VisuraAgent,
                 summa: SummaCraftAgent):
        super().__init__("DataMorph")
        self.tabu = tabu
        self.ctx = ctx
        self.vis = vis
        self.summa = summa

    def run(self, data: InputData, logs: Dict[str, Any] | None = None) -> str:
        if logs is None:
            logs = {}
        # Run agents conditionally
        if data.table:
            self.tabu.run(data, logs)
        if data.context:
            self.ctx.run(data, logs)
        if data.image_cues:
            self.vis.run(data, logs)
        # Compose summary
        summary = self.summa.run(data, logs)
        logs[self.name] = {"summary": summary}
        # Optionally return logs for external use
        return summary
