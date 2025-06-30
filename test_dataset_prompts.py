#!/usr/bin/env python3
"""
Test script to verify dataset-specific prompts are working correctly.
"""
from amaf.agents import ContextronAgent, TabuSynthAgent, VisuraAgent
from amaf.core import InputData
import json

def test_dataset_prompts():
    """Test that agents load dataset-specific prompts correctly."""
    
    # Test data
    test_data = InputData(
        context="Sample financial context for testing",
        table=[["Revenue", "2022", "2023"], ["Sales", "1000", "1200"]],
        image_path=[],
        user_profile="general"
    )
    
    # Test different datasets
    datasets = ["tatqa", "mmqa", "finqa", None]
    
    for dataset in datasets:
        print(f"\n=== Testing with dataset: {dataset or 'default'} ===")
        
        # Test ContextronAgent
        contextron = ContextronAgent(dataset=dataset)
        prompt_path = contextron.get_prompt_path("contextron.txt")
        print(f"Contextron prompt path: {prompt_path}")
        print(f"Prompt exists: {prompt_path.exists()}")
        
        # Test TabuSynthAgent
        tabusynth = TabuSynthAgent(dataset=dataset)
        prompt_path = tabusynth.get_prompt_path("tabu_synth.txt")
        print(f"TabuSynth prompt path: {prompt_path}")
        print(f"Prompt exists: {prompt_path.exists()}")
        
        # Test VisuraAgent
        visura = VisuraAgent(dataset=dataset)
        prompt_path = visura.get_prompt_path("visura.txt")
        print(f"Visura prompt path: {prompt_path}")
        print(f"Prompt exists: {prompt_path.exists()}")

if __name__ == "__main__":
    test_dataset_prompts() 