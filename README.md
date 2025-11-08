# The ML.ENERGY Benchmark

## Instructions

- Python setup (using `uv`)

```bash
# Install the project in a new uv-managed virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

- Data preparation
    * We use the audios in [FSD50K](https://zenodo.org/records/4060432) for our audio workload. There are two ways to download the audio files:
        1. Download them from Zenodo, the original source.
        ```
        pip install -q zenodo-get
        zenodo_get 10.5281/zenodo.4060432
        zip -F FSD50K.dev_audio.zip --out FSD50K.dev_audio_full.zip
        unzip FSD50K.dev_audio_full.zip
        ```
        2. Download from a [Hugging Face mirror](https://huggingface.co/datasets/Fhrozen/FSD50k) mirror. Note you might encounter rate limits why downloading, and setting your [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens) might be helpful
        ```
        huggingface-cli download Fhrozen/FSD50k \
          --repo-type dataset \
          --include "clips/dev/*"
        ```
        After downloading them, specify the path as the `--workload.audio-data-dir` argument.
    * We use the videos in [lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main) for our video workload. You need to extract the source files from the `tar.gz` files. We've prepared a helper script to download the compressed files from Hugging Face Hub and extract the files into a directory ready to use in the video workload. Note The entire dataset is about 1.2TB, and you need twice that space to extract. 
    ```
    python3 scripts/prepare_llava_videos.py <output_dir> --jobs 12
    ```
    Then specify the video path as the `--workload.video-data-dir` argument.

```bash
# This assumes the existence of extracted video/audio datasets.
python tests/llm/workloads.py
```

- Running the benchmark

```bash
# HF_TOKEN, HF_HOME, and CUDA_VISIBLE_DEVICES are required.
export HF_TOKEN=<your_hf_token>
export HF_HOME=<your_hf_home>

# Help and usage
python -m mlenergy.llm.benchmark --help
python -m mlenergy.llm.benchmark workload:image-chat --help

# LLM chat benchmark
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark --server-image vllm/vllm-openai:v0.11.1 --max-output-tokens 4096 workload:lm-arena-chat --workload.base-dir run/llm/lm-arena-chat/Qwen/Qwen3-8B/H100 --workload.model-id Qwen/Qwen3-8B --workload.num-requests 1024 --workload.gpu-model --workload.max-num-seqs 256

# LLM code completion benchmark
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark --server-image vllm/vllm-openai:v0.11.1 --max-output-tokens 1024 workload:sourcegraph-fim --workload.base-dir run/llm/sourcegraph-fim/google/codegemma-7b/H100 --workload.model-id google/codegemma-7b --workload.gpu-model --workload.num-requests 1024 --workload.max-num-seqs 256

# LLM problem solving benchmark
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark --server-image vllm/vllm-openai:v0.11.1 --max-output-tokens 10240 workload:gpqa --workload.base-dir run/llm/gpqa/Qwen/Qwen3-8B/H100 --workload.model-id Qwen/Qwen3-8B --workload.num-requests 256 --workload.gpu-model H100 --workload.max-num-seqs 128

# Multimodal image chat benchmark
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark --server-image vllm/vllm-openai:v0.9.2 workload:image-chat --workload.model-id Qwen/Qwen2.5-VL-7B-Instruct --workload.base-dir run/mllm/Qwen/Qwen2.5-VL-7B-Instruct --workload.num-requests 1000 --workload.num-images 1 --workload.max-num-seqs 64

# Input/Output length control
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark --server-image vllm/vllm-openai:v0.9.2 --ignore-eos workload:length-control --workload.model-id Qwen/Qwen2.5-VL-7B-Instruct --workload.base-dir run/mllm/Qwen/Qwen2.5-VL-7B-Instruct --workload.num-requests 1000 --workload.max-num-seqs 64 --workload.input-mean 500 --workload.output-mean 300

# Check the results
tree run
```

## Automated Job Generation

Instead of running benchmarks manually, use the config system to generate job scripts automatically.

### Directory Structure

**Task-level config:**
```
configs/vllm/{task}/benchmark.yaml
```

**Model+GPU-level config:**
```
configs/vllm/{task}/{org}/{model}/{gpu}/
  ├── monolithic.config.yaml
  ├── monolithic.env.yaml
  ├── num_gpus.txt
  ├── sweeps.yaml          # [Optional] Override sweep ranges
  ├── extra_body.json      # [Optional] Additional request parameters
  └── system_prompt.txt    # [Optional] System prompt for chat models
```

### Default Sweeps (`benchmark.yaml`)

```yaml
command_template: |
  python -m mlenergy.llm.benchmark \
    --workload.max-num-seqs {max_num_seqs}

sweep_defaults:
  - max_num_seqs: [8, 16, 32, 64, 96, 128, 192, 256]
```

### Custom Sweeps (`sweeps.yaml`)

Create `sweeps.yaml` in model+GPU dir to override defaults:

```yaml
# Narrow the range after initial exploration
sweep:
  - max_num_seqs: [64, 96, 128, 192, 256]
```

If `sweeps.yaml` exists, `sweep_defaults` is completely ignored.

### Generate Jobs

```bash
# Slurm
python scripts/generate_jobs.py generate-slurm \
  --output-dir slurm_jobs \
  --datasets lm-arena-chat \
  --gpu-models H100

# Pegasus
python scripts/generate_jobs.py generate-pegasus \
  --output-dir pegasus_queues
```

- Singularity

Convert into a Singularity image

```bash
singularity build vllm.sif docker://vllm/vllm-openai:v0.11.1
```

Running benchmarks with Singularity

```bash
# HF_TOKEN, HF_HOME, and CUDA_VISIBLE_DEVICES are required.
export HF_TOKEN=<your_hf_token>
export HF_HOME=<your_hf_home>

# Use --container-runtime singularity and specify the .sif file path
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark \
  --container-runtime singularity \
  --server-image /path/to/vllm.sif \
  --max-output-tokens 4096 \
  workload:lm-arena-chat \
  --workload.base-dir run/llm/lm-arena-chat/Qwen/Qwen3-8B/H100 \
  --workload.model-id Qwen/Qwen3-8B \
  --workload.num-requests 1024 \
  --workload.gpu-model H100 \
  --workload.max-num-seqs 256
```

Running vLLM server directly (for testing)

```bash
singularity exec --nv --env HF_TOKEN=hf_xxx --env PYTHONNOUSERSITE=1 vllm.sif vllm serve Qwen/Qwen3-4B
```
