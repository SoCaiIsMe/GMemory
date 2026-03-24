#!/bin/bash

if [ -f "./.env" ]; then
    export $(grep -v '^#' "./.env" | xargs)
fi

export HF_ENDPOINT=https://hf-mirror.com

# Options:
# --mas_memory:    empty, chatdev, metagpt, voyager, generative, memorybank, g-memory
# --mas_type:      autogen, dylan, macnet

# python3 tasks/run.py \
#     --task pddl \
#     --reasoning io \
#     --mas_memory g-memory \
#     --max_trials 30 \
#     --mas_type macnet \
#     --model Qwen/Qwen2.5-14B-Instruct \

# python3 tasks/run.py \
#     --task pddl \
#     --reasoning io \
#     --mas_memory g-memory \
#     --max_trials 30 \
#     --mas_type macnet \
#     --model deepseek-chat \

python3 tasks/run.py \
    --task pddl \
    --reasoning io \
    --mas_memory g-memory \
    --max_trials 30 \
    --mas_type macnet \
    --model Qwen/Qwen2.5-7B-Instruct \