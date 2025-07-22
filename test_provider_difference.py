#!/usr/bin/env python3
"""
Test script to verify which LLM provider is actually being used.
"""
import os
from amaf.agents.base import Agent

def test_provider():
    # Test with a prompt that should get different responses from different providers
    prompt = "Reply with exactly these words and nothing else: 'I am Mistral AI'"
    
    print("=== Testing Mistral Provider ===")
    if not os.getenv("MISTRAL_API_KEY"):
        print("❌ MISTRAL_API_KEY not set, skipping test")
        return
        
    try:
        agent = Agent("TestAgent", "test", provider="mistral", model="mistral-small-2506")
        response = agent._chat(prompt, temperature=0.0)
        print(f"Response: '{response.strip()}'")
        
        if "mistral" in response.lower():
            print("✅ Response suggests Mistral API was used")
        else:
            print("⚠️  Response doesn't clearly indicate Mistral")
            
    except Exception as e:
        print(f"❌ Error calling Mistral: {e}")

if __name__ == "__main__":
    test_provider() 