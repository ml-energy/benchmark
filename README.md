# The ML.ENERGY Benchmark

## Instructions

- [ ] Data preparation

```bash
python -m mlenergy.llm.workloads
```

- [ ] Running the benchmark

```bash
# HF_TOKEN, HF_HOME, and CUDA_VISIBLE_DEVICES are required.
export HF_TOKEN=<your_hf_token>
export HF_HOME=<your_hf_home>

# Help and usage
python -m mlenergy.llm.benchmark --help
python -m mlenergy.llm.benchmark workload:image-chat --help

# Example command
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark --max-num-seqs 512 --server-image vllm/vllm-openai:v0.9.2 --set-max-tokens workload:image-chat --workload.model-id Qwen/Qwen2.5-VL-7B-Instruct --workload.base-dir run/mllm/Qwen/Qwen2.5-VL-7B-Instruct --workload.num-requests 100 --workload.num-images 1
```
