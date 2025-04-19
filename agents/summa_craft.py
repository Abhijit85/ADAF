# SummaCraft agent implementation
from transformers import pipeline

summarizer = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

user_profile = "novice investor"
input_text = "The profit of Product B increased by 100% over the last quarter."
prompt = f"You're a financial advisor. Explain to a {user_profile}: {input_text}"

result = summarizer(prompt, max_new_tokens=100)[0]['generated_text']
print(result)
