# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="mradermacher/Llama-3-neoAI-8B-Chat-v0.1-i1-GGUF",
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "用中文介绍下你自己."}
  ],
  temperature=0.7,
)

print(completion.choices[0].message,flush=True)