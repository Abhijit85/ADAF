import os
from llama_api_client import LlamaAPIClient

client = LlamaAPIClient(
    api_key=os.environ.get("LLAMA_API_KEY"),  # This is the default and can be omitted
)

create_chat_completion_response = client.chat.completions.create(
    messages=[
        {
            "content": "How is the weather in Tokyo?",
            "role": "user",
        }
    ],
    model="Llama-3.3-8B-Instruct",
)
print(create_chat_completion_response.completion_message)