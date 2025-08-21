# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import json

def parse_output(response, stream=False, tool_functions=None):
    if stream:
        tool_args_str = ""
        chat_response = ""
        tool_name = None
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
            if delta.tool_calls:
                tool_call_delta = delta.tool_calls[0]
                if tool_call_delta.function.name:
                    tool_name = tool_call_delta.function.name
                if tool_call_delta.function.arguments:
                    tool_args_str += tool_call_delta.function.arguments
        print("\n")
        if tool_name:
            print(f"Function called: {tool_name}")
            print(f"Arguments: {tool_args_str}")
            result = tool_functions[tool_name](**json.loads(tool_args_str))
            print(f"Result: {result}")
    else:
        chat_response = response.choices[0].message.content
        print(chat_response)
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0].function
            print(f"Function called: {tool_call.name}")
            print(f"Arguments: {tool_call.arguments}")
            result = tool_functions[tool_call.name](**json.loads(tool_call.arguments))
            print(f"Result: {result}")
            return chat_response, tool_call, result
