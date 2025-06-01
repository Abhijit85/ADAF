# #!/usr/bin/env python3
# """CLI to run the AMAF summarization pipeline."""
# import json, argparse, sys, os
# from amaf.agents import (
#     TabuSynthAgent, ContextronAgent, VisuraAgent,
#     SummaCraftAgent, DataMorphAgent, InputData
# )

# def main():
#     parser = argparse.ArgumentParser(description="Run AMAF on a JSON input file.")
#     parser.add_argument("input", help="Path to input JSON file")
#     args = parser.parse_args()
#     with open(args.input, "r") as f:
#         raw = json.load(f)
#     data = InputData(**raw)
#     # Initialize agents
#     tabu = TabuSynthAgent()
#     ctx = ContextronAgent()
#     vis = VisuraAgent()
#     summa = SummaCraftAgent()
#     orchestrator = DataMorphAgent(tabu, ctx, vis, summa)
#     logs: dict = {}
#     summary = orchestrator.run(data, logs)
#     print("\n=== FINAL SUMMARY ===\n")
#     print(summary)
#     print("\n=== LOGS ===")
#     print(json.dumps(logs, indent=2))

# if __name__ == "__main__":
#     main()
# from amaf import DataMorphAgent, TabuSynthAgent, ContextronAgent, VisuraAgent,
#                  SummaCraftAgent, TrendAnalyserAgent, TopKFilterAgent, InputData
# import json, sys

# if __name__ == "__main__":
#     data = InputData(**json.load(open(sys.argv[1])))
#     logs = {}
#     agents = {
#         "TabuSynth": TabuSynthAgent(),
#         "Contextron": ContextronAgent(),
#         "Visura": VisuraAgent(),
#         "TrendAnalyser": TrendAnalyserAgent(),
#         "TopKFilter": TopKFilterAgent(),
#     }
#     orchestrator = DataMorphAgent(agents)
#     orchestrator.run(data, logs)
#     summary = SummaCraftAgent().run(data, logs).result
#     print("SUMMARY:\n", summary)

from amaf.agents import TabuSynthAgent, ContextronAgent, VisuraAgent, SummaCraftAgent, MCPController
registry = {
    "TabuSynth": TabuSynthAgent(),
    "Contextron": ContextronAgent(),
    "Visura": VisuraAgent(),
    "SummaCraft": SummaCraftAgent(),
}
controller = MCPController("protocols/basic.yaml", registry)
summary = controller.run(data, logs={})