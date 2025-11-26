# Overview

## Available Tasks

### LLM Tasks
- **lm-arena-chat**: Conversational prompts from LM Arena
- **gpqa**: Graduate-level problem solving questions with reasoning
- **sourcegraph-fim**: Fill-in-the-middle code completion

### MLLM Tasks
- **image-chat**: Conversational prompts that include images
- **video-chat**: Conversational prompts that include videos
- **audio-chat**: Conversational prompts that include audio clips

## Request Datasets

| **Task** | **Dataset** | **Auto-downloaded** |
|------|---------|-----------------|
| lm-arena-chat | LMArena Human Preference | Yes |
| gpqa | GPQA diamond | Yes |
| sourcegraph-fim | Sourcegraph FIM | Yes |
| image-chat | LMArena Vision Arena | Yes |
| video-chat | LLaVA-Video-178K | No (see [data-preparation.md](data-preparation.md)) |
| audio-chat | NVIDIA AudioSkills (FSD50K subset) | No (see [data-preparation.md](data-preparation.md)) |

## Runtime

### Container Runtime Support
- **Docker**: For cloud and local machines
- **Singularity**: For HPC clusters

### Environment Variables

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
