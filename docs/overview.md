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

We take model *inputs* from the following datasets for each task. Based on the input, we let the model generate their own outputs.

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


## Runtimes

The LLM/MLLM assumes that an underlying server provides the OpenAI Completions (for code completion) and Chat Completions (everything else) APIs. The server runs inside a container.
For diffusion models, we use the [xDiT](https://github.com/xdit-project/xDiT) runtime, and it does not run inside a container.

For the ML.ENERGY Benchmark 3.0 and the corresponding [leaderboard](https://ml.energy/leaderboard), we used the following runtime versions:

| **Model Architecture** | **Runtime** | **Version** |
|--------------------|---------|---------|
| LLM/MLLM | vLLM | 0.11.1 |
| Diffusion | xDiT | 0.4.5 |
