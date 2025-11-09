# Running Benchmarks

## Automated Job Generation (Recommended)

Generate batch jobs from configs for systematic benchmarking:

```bash
# Slurm
python scripts/generate_jobs.py generate-slurm \
  --output-dir slurm_jobs/ \
  --datasets lm-arena-chat gpqa \
  --gpu-models H100

# Pegasus
python scripts/generate_jobs.py generate-pegasus \
  --output-dir pegasus_queues/
```

Submit the generated jobs:
```bash
# Slurm
for job in slurm_jobs/*.sh; do sbatch "$job"; done

# Pegasus
pegasus submit pegasus_queues/1gpu.queue --gpus 1
```

### Configuration

Configs are in `configs/vllm/{task}/{org}/{model}/{gpu}/`:
- `monolithic.config.yaml`: vLLM server arguments
- `monolithic.env.yaml`: Environment variables
- `num_gpus.txt`: GPU count
- `sweeps.yaml` (optional): Override parameter sweep ranges

See `configs/vllm/{task}/benchmark.yaml` for command templates and default sweeps.

## Manual Execution

### LLM Tasks

**LM Arena Chat**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark \
  --server-image vllm/vllm-openai:v0.11.1 \
  --max-output-tokens 4096 \
  workload:lm-arena-chat \
  --workload.model-id Qwen/Qwen3-8B \
  --workload.base-dir run/llm/lm-arena-chat/Qwen/Qwen3-8B/H100 \
  --workload.num-requests 1024 \
  --workload.gpu-model H100 \
  --workload.max-num-seqs 256
```

**GPQA**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark \
  --server-image vllm/vllm-openai:v0.11.1 \
  --max-output-tokens 10240 \
  workload:gpqa \
  --workload.model-id Qwen/Qwen3-8B \
  --workload.base-dir run/llm/gpqa/Qwen/Qwen3-8B/H100 \
  --workload.num-requests 198 \
  --workload.gpu-model H100 \
  --workload.max-num-seqs 128
```

**Sourcegraph FIM**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark \
  --server-image vllm/vllm-openai:v0.11.1 \
  --max-output-tokens 1024 \
  workload:sourcegraph-fim \
  --workload.model-id google/codegemma-7b \
  --workload.base-dir run/llm/sourcegraph-fim/google/codegemma-7b/H100 \
  --workload.num-requests 1024 \
  --workload.gpu-model H100 \
  --workload.max-num-seqs 256
```

### MLLM Tasks

**Image Chat**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark \
  --server-image vllm/vllm-openai:v0.11.1 \
  --max-output-tokens 4096 \
  workload:image-chat \
  --workload.model-id Qwen/Qwen3-VL-8B-Instruct \
  --workload.base-dir run/mllm/image-chat/Qwen/Qwen3-VL-8B-Instruct/H100 \
  --workload.num-requests 1024 \
  --workload.num-images 1 \
  --workload.gpu-model H100 \
  --workload.max-num-seqs 64
```

**Video Chat**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark \
  --server-image vllm/vllm-openai:v0.11.1 \
  --max-output-tokens 4096 \
  workload:video-chat \
  --workload.model-id Qwen/Qwen3-VL-8B-Instruct \
  --workload.base-dir run/mllm/video-chat/Qwen/Qwen3-VL-8B-Instruct/H100 \
  --workload.num-requests 1024 \
  --workload.num-videos 1 \
  --workload.gpu-model H100 \
  --workload.max-num-seqs 64 \
  --workload.video-data-dir ${VIDEO_DATA_DIR}
```

**Audio Chat**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark \
  --server-image vllm-audio:v0.11.1 \
  --max-output-tokens 4096 \
  workload:audio-chat \
  --workload.model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --workload.base-dir run/mllm/audio-chat/Qwen/Qwen3-Omni-30B-A3B-Instruct/H100 \
  --workload.num-requests 1024 \
  --workload.num-audios 1 \
  --workload.gpu-model H100 \
  --workload.max-num-seqs 64 \
  --workload.audio-data-dir ${AUDIO_DATA_DIR}
```

### Using Singularity

Replace `--server-image` with path to `.sif` file and add `--container-runtime singularity`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark \
  --container-runtime singularity \
  --server-image /path/to/vllm.sif \
  workload:lm-arena-chat \
  --workload.model-id Qwen/Qwen3-8B \
  --workload.base-dir run/llm/lm-arena-chat/Qwen/Qwen3-8B/H100 \
  --workload.num-requests 1024 \
  --workload.gpu-model H100 \
  --workload.max-num-seqs 256
```

## Results

Results are saved to `{base_dir}/`:
- `requests.json`: Request data and workload config
- `results.json`: Benchmark results and metrics
- `driver_log.txt`: Client logs
- `server_log.txt`: vLLM server logs
