# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:4321/v1", api_key="dummy")

def get_weather(location: str, unit: str = "celsius"):
    return f"Getting the weather for {location} in {unit}..."
tool_functions = {"get_weather": get_weather}

demo_context = {
    "messages": [
        {"role": "user", "content": "You are a test system."},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Let me know the weather in Barcelona Spain."},
    ],
    "tools": [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                },
                "unit": {
                    "type": "string",
                    "description": "this is the unit of temperature"
                }
            },
            "required": [
                "location"
            ],
            "additionalProperties": False
        },
        "returns": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "description": "temperature in celsius"
                }
            },
            "required": [
                "temperature"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}],
    "add_generation_prompt": True
}


response = client.chat.completions.create(
    model="seed_oss",
    messages=demo_context["messages"],
    tools=demo_context["tools"],
    tool_choice="auto",
    max_tokens=4096,
    temperature=1.1,
    top_p=0.95,
)


chat_response = response.choices[0].message.content
print(chat_response)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {tool_functions[tool_call.name](**json.loads(tool_call.arguments))}")