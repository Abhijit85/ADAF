#!/usr/bin/env python3
"""
CLI: run AMAF via an MCP controller.
Usage:  python run_amaf.py <input.json>  [-p basic.yaml] [-d dataset]
"""
from amaf.agents import (
    TabuSynthAgent,
    ContextronAgent,
    VisuraAgent,
    SummaCraftAgent,
    TrendAnalyserAgent,
    MCPController,
    CalculatorAgent,
    PostCheckAgent
)
from amaf.core import InputData
import argparse
import json
import os
import sys
import pprint

from dotenv import load_dotenv
load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", help="AMAF-compatible input JSON")
    ap.add_argument("-p", "--protocol", 
                    default="basic.yaml",  # YAML file path
                    help="YAML protocol file for MCP")
    ap.add_argument("-d", "--dataset", 
                    default=None,
                    help="Dataset name (mmqa, tatqa, finqa) for dataset-specific prompts")
    
    args = ap.parse_args()

    # -------- read input
    try:
        with open(args.json_file, "r") as f:
            raw = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Cannot open {args.json_file}")

    data = InputData(**raw)

    # -------- build registry of agents
    registry = {
        "TabuSynth":   TabuSynthAgent(dataset=args.dataset),
        "Contextron":  ContextronAgent(dataset=args.dataset),
        "Visura":      VisuraAgent(dataset=args.dataset),
        "SummaCraft":  SummaCraftAgent(dataset=args.dataset),
        # add TrendAnalyser / TopKFilter if referenced in YAML
        "TrendAnalyser": TrendAnalyserAgent(dataset=args.dataset),
        "Calculator": CalculatorAgent(dataset=args.dataset),
        "PostCheck": PostCheckAgent(dataset=args.dataset),
    }

    # -------- run MCP
    controller = MCPController(args.protocol, registry, dataset=args.dataset)
    logs = {}
    summary = controller.run(data, logs)

    # -------- output
    print("\n=== FINAL SUMMARY ===\n")
    print(summary)
    print("\n=== LOGS (per-agent) ===")
    pprint.pp(logs)
    
    
if __name__ == "__main__":
    
    # Verify the OpenAI key is set
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    main()
