# Running Benchmarks

## Automated Job Generation (Recommended)

Generate batch jobs files from configs for systematic benchmarking. The script support Slurm and [Pegasus](https://github.com/jaywonchung/pegasus).

```bash
# Slurm
python scripts/generate_jobs.py generate-slurm \
  --output-dir slurm_jobs/ \
  --output.partition gpu \
  --output.account your_account \
  --output.cpus-per-gpu 10 \
  --output.mem-per-gpu 120G \
  --output.time-limit 1:00:00

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
  - max_num_seqs: [64, 128, 256]
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

Results are saved to `{base_dir}`, like `run/`. The hierarchical structure is as follows:

```
run/
├── llm/                              # Text-only tasks
│   └── {task}/                       # e.g., gpqa, lm-arena-chat, sourcegraph-fim
│       ├── requests/                 # Model-independent request data (shared)
│       │   └── {dataset_params}/    # e.g., num_requests+100+seed+48105
│       │       ├── requests.json    # Prompts, completions, multimodal content
│       │       └── multimodal_dump/ # [Optional] Dumped images/videos/audio
│       ├── tokenization/            # Model-dependent tokenization data
│       │   └── {model_id}/          # e.g., Qwen/Qwen3-8B
│       │       └── {dataset_params}.json  # Token counts and token IDs
│       └── results/                 # Benchmark results
│           └── {model_id}/
│               └── {gpu_model}/     # e.g., H100, B200
│                   └── {runtime_params}/  # e.g., num_requests+100+seed+48105+max_num_seqs+8+...
│                       ├── results.json      # Benchmark metrics and outputs
│                       ├── driver_log.txt    # Driver execution log
│                       ├── server_log.txt    # vLLM server log
│                       ├── requests.json@ -> ../../../requests/{dataset_params}/requests.json
│                       └── tokenization.json@ -> ../../../tokenization/{model_id}/{dataset_params}.json
│
└── mllm/                             # Multimodal tasks
    └── {task}/                       # e.g., image-chat, video-chat, audio-chat
        └── ...                       # Same structure as llm/

```

### Key Design Principles

1. **Data Deduplication**: `requests.json` is shared across all models and runtime configurations for the same task and dataset parameters, avoiding duplication of potentially large files (especially for video data).

2. **Tokenization Sharing**: Tokenization data is shared across different runtime configurations (e.g., different `max_num_seqs`) but regenerated when the model changes, since tokenization is model-dependent.

3. **Symlinks for Context**: Results directories contain symlinks to the corresponding `requests.json` and `tokenization.json` files, making it easy to understand which data was used without duplicating files.

4. **Hierarchical Organization**:
   - Task-level: Shared data that depends only on the task and dataset parameters
   - Model-level: Tokenization that depends on both task and model
   - Configuration-level: Results that depend on task, model, GPU, and runtime parameters

### File Contents

#### `requests.json`
Contains model-independent request data:
```json
{
  "data": [
    {
      "prompt": "What is the capital of France?",
      "completion": "The capital of France is Paris.",
      "multimodal_contents": [],
      "multimodal_content_paths": []
    }
  ],
  "dataset_params": {
    "num_requests": 100,
    "seed": 48105
  }
}
```

#### `tokenization.json`
Contains model-dependent tokenization data:
```json
{
  "tokenization": [
    {
      "prompt_len": 8,
      "expected_output_len": 7,
      "prompt_token_ids": [1234, 5678, ...]
    }
  ],
  "model_id": "Qwen/Qwen3-8B"
}
```

#### `results.json`
Contains benchmark results including timing, energy, and generation outputs:
```json
{
  "model_id": "Qwen/Qwen3-8B",
  "request_rate": 1.0,
  "results": [
    {
      "prompt": "...",
      "output_text": "...",
      "input_len": 8,
      "output_len": 7,
      "dataset_output_len": 7,
      "latency": 1.23,
      "ttft": 0.45,
      "itl": [0.1, 0.1, ...],
      "energy": 12.34
    }
  ],
  "metrics": { ... }
}
```
