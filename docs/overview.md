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

### Diffusion Tasks
- **text-to-image**: Image generation from text prompts
- **text-to-video**: Video generation from text prompts


## Request Datasets

| **Task** | **Dataset** | **Auto-downloaded** |
|------|---------|-----------------|
| lm-arena-chat | LMArena Human Preference | Yes |
| gpqa | GPQA diamond | Yes |
| sourcegraph-fim | Sourcegraph FIM | Yes |
| image-chat | LMArena Vision Arena | Yes |
| video-chat | LLaVA-Video-178K | No (see [data-preparation.md](data-preparation.md)) |
| audio-chat | NVIDIA AudioSkills (FSD50K subset) | No (see [data-preparation.md](data-preparation.md)) |
| text-to-image | Open Image Preferences | Yes |
| text-to-video | EvalCrafter T2V | Yes |
