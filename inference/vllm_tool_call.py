# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import argparse
from openai import OpenAI
from vllm_output_parser import parse_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--thinking_budget", type=int, default=-1)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens
    thinking_budget = args.thinking_budget
    stream = args.stream

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
        stream=stream,
        messages=demo_context["messages"],
        tools=demo_context["tools"],
        tool_choice="auto",
        max_tokens=max_new_tokens,
        temperature=1.1,
        top_p=0.95,
        extra_body={
            "chat_template_kwargs": {
                "thinking_budget": thinking_budget
            }
        }
    )

    parse_output(response, stream=stream, tool_functions=tool_functions)
