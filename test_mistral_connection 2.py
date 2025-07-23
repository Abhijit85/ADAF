"""Quick connectivity check for Mistral LLM.

Usage::

    export MISTRAL_API_KEY=sk-...
    python test_mistral_connection.py --model mistral-small-2506

If the call succeeds, the script prints the model response; otherwise, it prints the exception.
"""
import os
import argparse
from mistralai import Mistral


def main():
    parser = argparse.ArgumentParser(description="Ping Mistral chat endpoint")
    parser.add_argument("--model", default="mistral-small-2506", help="Mistral model name")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: MISTRAL_API_KEY environment variable not set")

    client = Mistral(api_key=api_key)

    try:
        resp = client.chat.complete(
            model=args.model,
            messages=[{"role": "user", "content": "Hello Mistral! Please reply with 'pong'."}],
            temperature=0.0,
        )
        print("✅ Mistral API call succeeded")
        print("Response:", resp.choices[0].message.content.strip())
    except Exception as e:
        print("❌ Mistral API call failed:")
        print(e)


if __name__ == "__main__":
    main() 