# The ML.ENERGY Benchmark

## Instructions

- [ ] Data preparation

```bash
python -m mlenergy.llm.workloads
```

- [ ] Running the benchmark

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=<your_hf_token>
export HF_HOME=<your_hf_home>
python -m mlenergy.llm.benchmark --help
python -m mlenergy.llm.benchmark workload:image-chat --help
```
