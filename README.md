---
title: ML.ENERGY Leaderboard
python_version: 3.9
app_file: app.py
sdk: gradio
sdk_version: 3.23.0
pinned: false
---

# ML.ENERGY Leaderboard

## Devs

Currently setup in `ampere02`:

1. Find model weights in `/data/leaderboard/weights/`, e.g. subdirectory `llama` and `vicuna`.

2. Let's share the Huggingface Transformer cache:

```bash
export TRANSFORMERS_CACHE=/data/leaderboard/hfcache
```

Run benchmarks like this:

```console
$ python benchmark.py --model-path /data/leaderboard/weights/lmsys/vicuna-7B --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json
$ python benchmark.py --model-path databricks/dolly-v2-12b --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json
```
