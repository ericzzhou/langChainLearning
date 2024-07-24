# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
  messages=[
    {"role": "system", "content": "你是一位专业顾问，你擅长中文交流，你的职责是负责为专注于亚洲零食在线销售的公司起名。"},
    {"role": "user", "content": "公司主打产品是亚洲零食，快消品"}
  ],
  temperature=0.7,
)

print(completion.choices[0].message)