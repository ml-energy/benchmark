#!/usr/bin/env bash

run() {
  # TP 4 GPUs
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 5.00 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 4.50 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 4.00 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 3.50 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 3.00 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 2.50 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 2.00 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 1.50 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 1.00 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.75 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.50 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.25 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3

  # TP 2 GPUs
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 2.50 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 2.25 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 2.00 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 1.75 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 1.50 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 1.25 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 1.00 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.75 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.50 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.25 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1

  # 1 GPU
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 1.25 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 1.125 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 1.00 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.875 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.75 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.625 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.50 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.375 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.25 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
  python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 0.125 --power-limit $PL --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0
}

# Warmup
timeout --signal SIGINT 120 python scripts/benchmark_one.py --backend vllm --server-image mlenergy/vllm:v0.3.0-openai --model meta-llama/Llama-2-13b-chat-hf --sharegpt-path ../../../sharegpt/ShareGPT_V3_filtered_500.json --request-rate 5.00 --power-limit 300 --result-root results/2024-02-19-scaling --huggingface-token $HF_TOKEN --gpu-ids 0 1 2 3

# PL=300
# run
#
# PL=275
# run
#
# PL=250
# run
#
# PL=225
# run
#
# PL=200
# run

PL=175
run

PL=150
run

PL=125
run

PL=100
run
