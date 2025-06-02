from amaf.agents.base import Agent
from amaf.core import AgentOutput


class DataMorphAgent(Agent):
    def __init__(self, agents):
        super().__init__("DataMorph")
        self.agents = agents  # dict[str, Agent]

    def run(self, data, logs):
        # 1) meta‑analysis
        row_cnt = len(data.table.get("rows", []))
        if row_cnt > 40:
            self.agents["TopKFilter"].run(data, logs)  # focus summary

        # always run base trio
        for key in ("TabuSynth", "Contextron", "Visura"):
            logs[key] = self.agents[key].run(data, logs).__dict__

        # optional extra trend analysis if user is analyst
        if data.user_profile == "analyst":
            logs["TrendAnalyser"] = self.agents["TrendAnalyser"].run(data, logs).__dict__

        return AgentOutput(cot="(controller)", result="")
