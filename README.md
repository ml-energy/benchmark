---
title: "ML.ENERGY Leaderboard"
emoji: "⚡"
python_version: "3.9"
app_file: "app.py"
sdk: "gradio"
sdk_version: "3.39.0"
pinned: true
tags: ["energy", "leaderboard"]
---

# ML.ENERGY Leaderboard

[![Leaderboard](https://custom-icon-badges.herokuapp.com/badge/ML.ENERGY-Leaderboard-blue.svg?logo=ml-energy-2)](https://ml.energy/leaderboard)
[![Deploy](https://github.com/ml-energy/leaderboard/actions/workflows/push_spaces.yaml/badge.svg?branch=web)](https://github.com/ml-energy/leaderboard/actions/workflows/push_spaces.yaml)
[![Apache-2.0 License](https://custom-icon-badges.herokuapp.com/github/license/ml-energy/leaderboard?logo=law)](/LICENSE)

How much energy do GenAI models like LLMs and Diffusion models consume?

This README focuses on explaining how to run the benchmark yourself.
The actual leaderboard is here: https://ml.energy/leaderboard.

## Repository Organization

```
leaderboard/
├── benchmark/      # Benchmark scripts & instructions
├── data/           # Benchmark results
├── deployment/     # Colosseum deployment files
├── spitfight/      # Python package for the Colosseum
├── app.py          # Leaderboard Gradio app definition
└── index.html      # Embeds the leaderboard HuggingFace Space
```

## Colosseum

We instrumented [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) so that it measures and returns GPU energy consumption.
Then, our [controller](/spitfight/colosseum/controller) server receives user prompts from the [Gradio app](/app.py), selects two models randomly, and streams model responses back with energy consumption.

## Running the Benchmark

We open-sourced the entire benchmark with instructions here: [`./benchmark`](./benchmark)

## Citation

Please refer to our BibTeX file: [`citation.bib`](/docs/citation.bib).
