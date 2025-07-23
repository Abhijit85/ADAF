from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  model="deepseek/deepseek-r1:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ],
  #temprature = 0.2
)
print(completion.choices[0].message.content)