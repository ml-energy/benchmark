The goal of the ML.ENERGY Leaderboard is to give people a sense of how much **energy** LLMs would consume.

## How is energy different?

Even between models with the exact same architecture and size, the average energy consumption per prompt is different because they have **different verbosity**.
That is, when asked the same thing, they answer in different lengths.

## Metrics

- `gpu`: NVIDIA GPU model name
- `task`: Name of the task. See *Tasks* below for details.
- `throughput` (token/s): The average number of tokens generated per second.
- `response_length` (token): The average number of tokens in the model's response.
- `latency` (s): The average time it took for the model to generate a response.
- `energy` (J): The average energy consumed by the model to generate a response.
- `parameters`: The number of parameters the model has, in units of billion.

## Tasks

For each task, every model uses the same system prompt. We still account for differences in roles, e.g. `USER`, `HUMAN`, `ASSISTANT`, `GPT`.

| Name | System prompt |
|--|--|
| chat | A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. |
| chat-concise | A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant's answers are very concise. |
| instruct | Below is an instruction that describes a task. Write a response that appropriately completes the request. |
| instruct-concise | Below is an instruction that describes a task. Write a response that appropriately completes the request. The response should be very concise. |

## Setup

Find our benchmark script for one model [here](https://github.com/ml-energy/leaderboard/blob/master/benchmark.py).

### Software

- PyTorch 2.0.1
- [FastChat](https://github.com/lm-sys/fastchat) -- For various model support
- [Zeus](https://ml.energy/zeus) -- For GPU energy measurement

### Hardware

- NVIDIA A40 GPU

### Parameters

- Model
  - Batch size 1
  - FP16
- Sampling (decoding)
  - Greedy sampling from multinomial distribution
  - Temperature 0.7
  - Repetition penalty 1.0

## Data

We randomly sampled around 3000 prompts from the [cleaned ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
See [here](https://github.com/ml-energy/leaderboard/tree/master/sharegpt) for more detail on how we created the benchmark dataset.

We used identical system prompts for all models (while respecting their own *role* tokens):
```
A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
```

## Upcoming

- Compare against more optimized inference runtimes, like TensorRT.
- Other GPUs
- Other model/sampling parameters
- More models
- Model quality evaluation numbers (e.g., AI2 Reasoning Challenge, HellaSwag)

# License

This leaderboard is a research preview intended for non-commercial use only.
The use of LLaMA weights are subject to their [license](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).
Please direct inquiries and reports of potential license/copyright violation to Jae-Won Chung.
