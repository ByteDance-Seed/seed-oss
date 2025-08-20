# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:4321/v1", api_key="dummy")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "How to make pasta?"}],
    max_tokens=2048,
    temperature=1,
    top_p=0.95,
)

chat_response = response.choices[0].message.content
print(chat_response)
