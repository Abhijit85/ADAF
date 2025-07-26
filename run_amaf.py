#!/usr/bin/env python3
"""
CLI: run AMAF via an MCP controller.
Usage:  python run_amaf.py <input.json>  [-p basic.yaml] [-d dataset] [--provider openai|mistral|gemini] [--model model_name]
"""
from amaf.agents import (
    TabuSynthAgent,
    ContextronAgent,
    VisuraAgent,
    SummaCraftAgent,
    TrendAnalyserAgent,
    MCPController
)
from amaf.core import InputData
import argparse
import json
import os
import sys
import pprint
import time

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
    ap.add_argument("--provider", choices=["openai", "mistral", "gemini","lambda"], default="openai", help="LLM provider to use")
    ap.add_argument("--model", default="gpt-3.5-turbo", help="Model name to use with the provider")
    ap.add_argument('--method', type=str, default=None, help='Prompting method/folder')
    
    args = ap.parse_args()

    # ---- provider-specific API key sanity check ----
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set for provider openai")
    if args.provider == "mistral" and not os.getenv("MISTRAL_API_KEY"):
        sys.exit("MISTRAL_API_KEY not set for provider mistral")
    if args.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set for provider gemini")
    if args.provider == "lambda" and not os.getenv("LAMBDA_API_KEY"):
        sys.exit("LAMBDA_API_KEY not set for provider lambda")

    # -------- read input
    try:
        with open(args.json_file, "r") as f:
            raw = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Cannot open {args.json_file}")

    # Handle both 'question' (singular) and 'questions' (plural) for compatibility
    if 'question' in raw and 'questions' not in raw:
        raw['questions'] = [raw['question']]
        del raw['question']
    elif 'question' in raw and 'questions' in raw:
        # If both exist, prefer 'questions' but warn
        print(f"Warning: Both 'question' and 'questions' found in {args.json_file}. Using 'questions'.")
        del raw['question']

    data = InputData(**raw)

    # Print provider / model debug line once per run
    print(f"[RUN_AMAF] provider={args.provider} | model={args.model}")

    # -------- build registry of agents
    registry = {
        "TabuSynth":   TabuSynthAgent(dataset=args.dataset, provider=args.provider, model=args.model, method=args.method),
        "Contextron":  ContextronAgent(dataset=args.dataset, provider=args.provider, model=args.model, method=args.method),
        "Visura":      VisuraAgent(dataset=args.dataset, provider=args.provider, model=args.model, method=args.method),
        "SummaCraft":  SummaCraftAgent(dataset=args.dataset, provider=args.provider, model=args.model, method=args.method),
        "TrendAnalyser": TrendAnalyserAgent(dataset=args.dataset, provider=args.provider, model=args.model, method=args.method),
    }

    # -------- run MCP with retry logic
    controller = MCPController(args.protocol, registry, dataset=args.dataset, provider=args.provider, model=args.model)
    logs = {}
    
    # Retry logic for rate limiting
    max_retries = 5
    base_sleep = 20  # Start with 20 seconds
    summary = None
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} for {args.json_file}")
            summary = controller.run(data, logs)
            print(f"✅ Success on attempt {attempt + 1}")
            break  # Success!
            
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = ("rate limit" in error_msg or 
                           "429" in error_msg or 
                           "rate_limited" in error_msg)
            
            if is_rate_limit and attempt < max_retries - 1:
                sleep_time = base_sleep * (2 ** attempt)  # Exponential backoff: 20s, 40s, 80s, 160s
                print(f"⚠️  Rate limit hit on attempt {attempt + 1}. Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            else:
                print(f"❌ Failed after {attempt + 1} attempts: {e}")
                if is_rate_limit:
                    print("Rate limit exceeded after all retries. Skipping this example.")
                raise  # Re-raise the exception to stop processing this example
    
    if summary is None:
        print(f"❌ All retries failed for {args.json_file}")
        return  # Don't write output file

    # -------- output
    print("\n=== FINAL SUMMARY ===\n")
    print(summary)
    print("\n=== LOGS (per-agent) ===")
    pprint.pp(logs)
    
    
if __name__ == "__main__":
    main()
