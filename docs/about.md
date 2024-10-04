The goal of the ML.ENERGY Leaderboard is to give people a sense of how much **energy** LLMs would consume, and the complex tradeoffs between energy, system performance, and user experience.

The code for the leaderboard, backing data, and scripts for benchmarking are all open-source in our [repository](https://github.com/ml-energy/leaderboard).
We'll see you at the [Discussion board](https://github.com/ml-energy/leaderboard/discussions), where you can ask questions, suggest improvement ideas, or just discuss leaderboard results!

## LLM Text Generation Benchmark

This category includes LLM Chat, LLM Code, and VLM Visual Chat.

### Software
- CUDA 12.4
- [vLLM](https://github.com/vllm-project/vllm) 0.5.4 -- For inference serving
- [Zeus](https://ml.energy/zeus) -- For GPU time and energy measurement

### Hardware
- NVIDIA A100-SXM4-40GB GPU (AWS p4d.24xlarge)
- NVIDIA H100 80GB HBM3 GPU (AWS p5.48xlarge)

### Data

| Task | Dataset |
| -------------- | --------------- |
| LLM Chat | 500 human prompts from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| LLM Code | HumanEval+ from [EvalPlus](https://github.com/evalplus/evalplus) |
| VLM Visual Chat | 500 image and prompt pairs from the [LLaVA instruction dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |

### Benchmarking process

We are interested in measuring the *steady-state* of the online serving system, excluding ramp up periods (when the server is gradually loaded with requests) and cooldown (when the server is draining the queue) periods.
Therefore, we submit all the requests at the beginning of our benchmark while limiting the serving system's maximum batch size to create a steady-state serving system.
The steady-state finishes when the serving system's queue length reaches zero, and we collect the timing and energy consumption of each batch during the steady-state to derive our metrics.
The maximum batch size configuration is varied in order to change the system's utilization.
Beyond a certain maximum batch size, the actual batch size does not increase due to memory constraints, which means the system will be overloaded, and we stop increasing the maximum batch size.

## Diffusion Benchmark

This category includes Diffusion text to image, text to video, and image to video.

### Software
- CUDA 12.4
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) 0.29.2 -- For inference
- [Zeus](https://ml.energy/zeus) -- For GPU time and energy measurement

### Hardware
- NVIDIA A100-SXM4-40GB GPU (AWS p4d.24xlarge)
- NVIDIA H100 80GB HBM3 GPU (AWS p5.48xlarge)

### Data

| Task | Dataset |
| -------------- | --------------- |
| Text to image | Prompts from [PartiPrompts](https://huggingface.co/datasets/nateraw/parti-prompts) |
| Text to video | Captions from [ShareGPT4Video](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video) |
| Image to video | Caption and first frame pairs from [ShareGPT4Video](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video) |

### Benchmarking process

Since Diffusion model computations are more or less the same regardless of the input, we sample batches from each dataset and run them back-to-back to obtain stable measurements.
The batch size is increased (in powers of two) until the GPU runs out of memory.

## The ML.ENERGY Initiative

Are you interested in learning more about our ML energy measurement & optimization works?
Meet us at the [**ML.ENERGY Initiative**](https://ml.energy) homepage!

---

## Acknowledgements

> Any opinions, findings, and conclusions of our works are those of the author(s) and do not necessarily represent the official policy of any of these funding organizations.

We thank the Mozilla Foundation for funding us via the [2024 Mozilla Technology Fund](https://foundation.mozilla.org/en/blog/open-source-AI-for-environmental-justice).

## License

This leaderboard is a research preview intended for non-commercial use only.
Model weights were taken as is from the Hugging Face Hub if available and are subject to their licenses.
Please direct inquiries/reports of potential violation to Jae-Won Chung.

## Contact

Please direct general questions and issues related to the leaderboard to our GitHub repository's [discussion board](https://github.com/ml-energy/leaderboard/discussions).
You can find the ML.ENERGY initiative members in [our homepage](https://ml.energy#members).
If you need direct communication, please email admins@ml.energy.
