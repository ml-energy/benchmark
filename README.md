# The ML.ENERGY Benchmark

## Instructions

- System setup

```bash
# Increase the number of max open files
ulimit -n 10000
```

- Python setup

```bash
# Install the project in a new uv-managed virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

- Data preparation

**TODO(Jeff)**: Add instructions for downloading and extracting video/audio datasets manually.

```bash
# This assumes the existence of extracted video/audio datasets.
python -m mlenergy.llm.workloads
```

- Running the benchmark

```bash
# HF_TOKEN, HF_HOME, and CUDA_VISIBLE_DEVICES are required.
export HF_TOKEN=<your_hf_token>
export HF_HOME=<your_hf_home>

# Help and usage
python -m mlenergy.llm.benchmark --help
python -m mlenergy.llm.benchmark workload:image-chat --help

# Example command
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark --server-image vllm/vllm-openai:v0.9.2 workload:image-chat --workload.model-id Qwen/Qwen2.5-VL-7B-Instruct --workload.base-dir run/mllm/Qwen/Qwen2.5-VL-7B-Instruct --workload.num-requests 1000 --workload.num-images 1 --workload.max-num-seqs 64

# Check the results
tree run
```
