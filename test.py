import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
print("api_base :", getattr(openai, "api_base", "default"))
print("api_type :", getattr(openai, "api_type", "openai (default)"))
print("api_version :", getattr(openai, "api_version", ""))
print("model :", os.getenv("AMAF_MODEL"))
try:
    openai.chat.completions.create(
        model=os.getenv("AMAF_MODEL", "gpt-3.5-turbo-1106"),
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1
    )
    print("Model works ✅")
except Exception as e:
    print("Still invalid:", e)