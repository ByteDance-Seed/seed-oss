# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import List


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str or List, default=None)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["flash_attention_2", "eager", "sdpa"])
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--thinking_budget", type=int, default=-1)
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    parser.add_argument("--load_in_4bit", type=bool, default=False)
    args = parser.parse_args()

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else model_path
    use_cache = args.use_cache
    max_new_tokens = args.max_new_tokens
    thinking_budget = args.thinking_budget
    load_in_8bit = args.load_in_8bit
    load_in_4bit = args.load_in_4bit

    if load_in_4bit or load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, bnb_4bit_compute_dtype=torch.float16
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="auto",
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model.eval()

    if args.prompts is None:
        prompts = [
            [{"role": "user", "content": "How to make pasta?"}],
            [{"role": "user", "content": "Please write the quick sort algorithm in Python."}]
        ]  
    else:
        if isinstance(args.prompts, str):
            args.prompts = [args.prompts]
        prompts = [
            [{"role": "user", "content": prompt}] for prompt in args.prompts
        ]

    for messages in prompts:
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,              
            add_generation_prompt=True,
            return_tensors="pt",
            thinking_budget=thinking_budget
        )

        tokenized_chat = tokenized_chat.to(model.device)
        generated_id = model.generate(
            tokenized_chat, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache
        )
        output = tokenizer.decode(generated_id[0], skip_special_tokens=True)

        print(output)

    del model
    torch.cuda.empty_cache()