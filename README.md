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

# Example command
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark --server-image vllm/vllm-openai:v0.9.2 workload:image-chat --workload.model-id Qwen/Qwen2.5-VL-7B-Instruct --workload.base-dir run/mllm/Qwen/Qwen2.5-VL-7B-Instruct --workload.num-requests 1000 --workload.num-images 1 --workload.max-num-seqs 64

# Input/Output length control
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.llm.benchmark --server-image vllm/vllm-openai:v0.9.2 --ignore-eos workload:length-control --workload.model-id Qwen/Qwen2.5-VL-7B-Instruct --workload.base-dir run/mllm/Qwen/Qwen2.5-VL-7B-Instruct --workload.num-requests 1000 --workload.max-num-seqs 64 --workload.input-mean 500 --workload.output-mean 300

# Prefill-Decode disaggregated serving
## 3P1D
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m mlenergy.llm.benchmark --overwrite-results --endpoint-type openai --server-image vllm/vllm-openai:v0.10.0 workload:gpqa --workload.model-id meta-llama/Llama-3.1-8B-Instruct --workload.base-dir run/llm/meta-llama/Llama-3.1-8B-Instruct --workload.num-requests 1000 --workload.max-num-seqs 64 --workload.num-prefills 3 --workload.num-decodes 1

## TP=4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m mlenergy.llm.benchmark --overwrite-results --endpoint-type openai --server-image vllm/vllm-openai:v0.10.0 workload:gpqa --workload.model-id meta-llama/Llama-3.1-70B-Instruct --workload.base-dir run/llm/meta-llama/Llama-3.1-70B-Instruct --workload.num-requests 200 --workload.max-num-seqs 64 --workload.num-prefills 1 --workload.num-decodes 1

# Check the results
tree run
```
