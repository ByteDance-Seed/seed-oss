# !/bin/bash

python3 -m vllm.entrypoints.openai.api_server \
    --host localhost \
    --port 4321 \
    --enable-auto-tool-choice \
    --tool-call-parser seed_oss \
    --trust-remote-code \
    --model ./Seed-OSS-36B-Instruct \
    --chat-template ./Seed-OSS-36B-Instruct/chat_template.jinja \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --served-model-name seed_oss