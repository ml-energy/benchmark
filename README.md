# The ML.ENERGY Benchmark

Benchmarking framework for measuring energy consumption and performance of Large Language Models (LLMs) and Multimodal LLMs (MLLMs).

## Quick Start

```bash
# Install
git clone https://github.com/ml-energy/leaderboard.git
cd leaderboard
uv sync
source .venv/bin/activate
```

- Diffusion benchmark setup

```bash
uv venv --python=3.12
source .venv/bin/activate

# Hopper
uv pip install packaging einops ninja wheel psutil && \
uv pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu126 && \
uv pip install flash-attn==2.7.0.post2 --no-build-isolation && \
uv pip install xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu126 && \
uv pip install .[diffusion]
```

```bash
# Blackwell
uv pip install packaging einops ninja wheel psutil && \
uv pip install "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128 && \
uv pip install flash-attn==2.7.4.post1 --no-build-isolation && \
uv pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128 && \
uv pip install .[diffusion]

```

``

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
