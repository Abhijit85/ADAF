# Contextron agent implementation
from transformers import pipeline

context_aware_pipe = pipeline("text2text-generation", model="google/flan-t5-xl")
surrounding_text = "This report summarizes quarterly profit changes due to inflation."
query = f"Summarize the effect of inflation based on this note: {surrounding_text}"

print(context_aware_pipe(query)[0]['generated_text'])
