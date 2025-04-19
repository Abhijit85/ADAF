# TabuSynth agent implementation
from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")

data = {'Product': ['A', 'B'], 'Q4 Profit': ['$100K', '$200K']}
table = pd.DataFrame.from_dict(data)
queries = ["What is the Q4 Profit of Product B?"]

inputs = tokenizer(table=table, queries=queries, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
answer_coordinates, _ = tokenizer.convert_logits_to_predictions(inputs, logits)
