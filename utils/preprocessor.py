# Preprocessing utilities
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def preprocess(example):
    prompt = f"Summarize table with context: {example['context']}. Table: {example['table']}. Profile: {example['user_profile']}"
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(example['summary'], truncation=True, padding="max_length", max_length=128)
    return {"input_ids": inputs['input_ids'], "labels": outputs['input_ids']}

dataset = load_dataset("your/finance_dataset")
tokenized = dataset.map(preprocess)
