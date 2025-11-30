import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 让 .env 生效

key = os.environ.get("OPENAI_API_KEY")
print("OPENAI_API_KEY is None? ->", key is None)
print("OPENAI_API_KEY 前 10 个字符 ->", key[:10] if key else None)

client = OpenAI(api_key=key)

resp = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "测试一下，你在吗？"},
    ],
    max_tokens=20,
)

print("API 返回：", resp.choices[0].message.content)
