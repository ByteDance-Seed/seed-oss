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

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        stream=stream,
        messages=[{"role": "user", "content": "How to make pasta?"}],
        max_tokens=max_new_tokens,
        temperature=1.1,
        top_p=0.95,
        extra_body={
            "chat_template_kwargs": {
                "thinking_budget": thinking_budget
            }
        }
    )

    parse_output(response, stream=stream)
