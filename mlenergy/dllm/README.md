# Diffusion LLM

## Benchmark Description
### Task Benchmark
- Chat: Text-based conversational AI applications, like ChatGPT or Claude
    - Each batch of chat conversations has a fixed output length

### Control Benchmark
- Generation Length
- Block Length
- Cache mode (dual or prefix)
- Remasking strategy (low confidence remasking or random remasking)

## Installation
We use a fork of Fast-dLLM that makes the import easier. The fork is available as a submodule. To install it:
```
git submodule update --init
cd third-party/Fast-dLLM
uv pip install -e .
```

In the above installation script, there are also functions that subsitute the import headers of the example applications of Fast-dLLMs, which might be helpful.

## Usage Examples
We currently support LLaDA and Dream models. **Dream can only run with batch size of 1.**
```
CUDA_VISIBLE_DEVICES=0 python -m mlenergy.dllm.benchmark \
  dllm-runtime:fast-dllm-runtime \
  --dllm-runtime.steps 128 \
  --dllm-runtime.gen-length 128 \
  --dllm-runtime.block-length 32 \
  --dllm-runtime.remasking low_confidence
```