from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama3-70b-instruct",
  messages=[{"role":"user","content":"Provide me a article on AI"}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")