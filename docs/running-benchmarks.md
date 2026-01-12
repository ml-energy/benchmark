# Running Benchmarks

## Environment Setup

Generally NVIDIA GPUs are assumed, but with runtimes compatible with other platforms (e.g., vLLM on ROCm), the benchmark should run fine.
For energy measurement support, see [Zeus's documentation](https://ml.energy/zeus/measure/#hardware-support).

Clone the repository:

```bash
git clone https://github.com/ml-energy/leaderboard.git
```

Python virtual environment (using [`uv`](https://astral.sh/uv)):

```bash
cd benchmark

uv sync
source .venv/bin/activate
```

Extra depedencies for the diffusion benchmark:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

uv pip install packaging einops ninja wheel psutil && \
uv pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128 && \
uv pip install flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
uv pip install xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128 && \
uv pip install .[diffusion]

```

## Environment Variables

Required for all tasks:
```bash
export HF_TOKEN="your_huggingface_token"
export HF_HOME="/path/to/huggingface/cache"
export CUDA_VISIBLE_DEVICES=0
```

Required for MLLM tasks:
```bash
export VIDEO_DATA_DIR="/path/to/llava-video-178k"  # For video-chat
export AUDIO_DATA_DIR="/path/to/fsd50k"            # For audio-chat
```

## Automated Job Generation

Generate batch jobs files from configs for systematic benchmarking. The script support Slurm and [Pegasus](https://github.com/jaywonchung/pegasus).

```bash
# Slurm
python scripts/generate_jobs.py generate-slurm \
  --output-dir slurm_jobs/ \
  --output.partition gpu \
  --output.account your_account \
  --output.cpus-per-gpu 10 \
  --output.mem-per-gpu 120G \
  --output.time-limit 3:00:00

# Pegasus
python scripts/generate_jobs.py generate-pegasus \
  --output-dir pegasus_jobs/ \
  --output.gpus-per-node 8 \
  --output.hostname localhost
```

You can generate only a subset of the jobs by specifying filters:

```bash
python scripts/generate_jobs.py generate-slurm \
  --output-dir slurm_jobs/ \
  --datasets lm-arena-chat gpqa \
  --gpu-models H100 \
  ...
```

You can submit jobs so Slurm as you normally would:

```bash
for file in slurm_jobs/*8gpu*; do
  sbatch $file
done
```

And also Pegasus (install with `cargo install pegasus-ssh`):

```bash
pegasus q --hosts-file pegasus_jobs/hosts_8gpu.yaml --queue-file pegasus_jobs/queue_8gpu.yaml
```


### Configuration

Configs are in `configs/vllm/{task}/{org}/{model}/{gpu}/`:
- `monolithic.config.yaml`: vLLM server arguments
- `monolithic.env.yaml`: Environment variables
- `num_gpus.txt`: GPU count
- `sweeps.yaml` (optional): Override parameter sweep ranges

See `configs/vllm/{task}/benchmark.yaml` for command templates and default sweeps.

### Templating System

Benchmark commands use a two-level templating system:

**Built-in parameters** (automatically filled from config structure):
- `{model_id}`: Model identifier from directory structure (e.g., `Qwen/Qwen3-8B`)
- `{gpu_model}`: GPU type from directory structure (e.g., `H100`)
- `{num_gpus}`: Number of GPUs from `num_gpus.txt`

**Generator-level parameters** (specified when generating jobs):
- `{container_runtime}`: Container runtime (`docker` or `singularity`)
- `{server_image}`: Container image path (Docker image or `.sif` file)

**Sweep parameters** (defined in `sweep_defaults` or `sweeps.yaml`):
- `{max_num_seqs}`: Batch size parameter (example)
- Any custom parameters you add to sweeps

```yaml
# sweeps.yaml example
sweep:
  - max_num_seqs: [64, 128, 256]  # Applies to all #GPUs
  - num_gpus: [2]                 # Applies only when running with 2 GPUs
    max_num_seqs: [384, 512]
```

### Container Runtimes

The benchmark supports Docker and Singularity (e.g., for Docker-less HPC environments).

For Singularity, you can build a Singularity image (`.sif` file) from a Docker image:

```bash
singularity build vllm.sif docker://vllm/vllm-openai:v0.11.1
```

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

See [Analyzing Results](analyzing-results.md) for details on result organization, file formats, and analysis techniques.
