# The ML.ENERGY Benchmark

Benchmarking framework for measuring energy consumption and performance of Large Language Models (LLMs) and Multimodal LLMs (MLLMs).

## Quick Start

```bash
# Install
git clone https://github.com/ml-energy/leaderboard.git
cd leaderboard
uv sync
source .venv/bin/activate

- Diffusion benchmark setup

```bash
uv venv --python=3.12
source .venv/bin/activate

uv pip install packaging einops ninja wheel psutil && \
uv pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu126 && \
uv pip install flash-attn==2.7.0.post2 --no-build-isolation && \
uv pip install xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu126 && \
uv pip install .[diffusion]
```

- Data preparation

**TODO(Jeff)**: Add instructions for downloading and extracting video/audio datasets manually.

```bash
# This assumes the existence of extracted video/audio datasets.
python -m mlenergy.llm.workloads
# Setup
export HF_TOKEN="your_huggingface_token"
export HF_HOME="/path/to/huggingface/cache"
export CUDA_VISIBLE_DEVICES=0

# Run
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark \
  --server-image vllm/vllm-openai:v0.11.1 \
  --max-output-tokens 4096 \
  workload:lm-arena-chat \
  --workload.model-id Qwen/Qwen3-8B \
  --workload.base-dir run/llm/lm-arena-chat/Qwen/Qwen3-8B/H100 \
  --workload.num-requests 128 \
  --workload.gpu-model H100 \
  --workload.max-num-seqs 64
```

## Documentation

- **[Overview](docs/overview.md)**: Tasks, datasets, runtime
- **[Data Preparation](docs/data-preparation.md)**: Dataset download scripts
- **[Running Benchmarks](docs/running-benchmarks.md)**: Job generation and manual execution

## Development

```bash
# Lint and type check
./scripts/lint.sh

# Test
pytest
```
