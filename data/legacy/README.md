# Data files for the ML.ENERGY Leaderboard

This directory holds all the data for the leaderboard table.

Code that reads in the data here can be found in the constructor of `TableManager` in `app.py`.

## Parameters

There are two types of parameters: (1) Those that become radio buttons on the leaderboard and (2) those that become columns on the leaderboard table.
Models are always placed in rows.

Currently, there are only two parameters that become radio buttons: GPU model (e.g., V100, A40, A100) and task (e.g., chat, chat-concise, instruct, and instruct-concise).
This is defined in the `schema.yaml` file.

Radio button parameters have their own CSV file in this directory.
For instance, benchmark results for the *chat* task ran on an *A100* GPU lives in `A100_chat_benchmark.csv`. This file name is dynamically constructed by the leaderboard Gradio application by looking at `schema.yaml` and read in as a Pandas DataFrame.

Parameters that become columns in the table are put directly in the benchmark CSV files, e.g., `batch_size` and `datatype`.

## Adding new models

1. Add your model to `models.json`.
   - The model's JSON key should be its unique codename, e.g. Hugging Face Hub model name. It's usually not that readable.
   - `url` should point to a page where people can obtain the model's weights, e.g. Hugging Face Hub.
   - `nickname` should be a short human-readable string that identifies the model.
   - `params` should be an integer rounded to billions.
  
1. Add NLP dataset evaluation scores to `score.csv`.
   - `model` is the model's JSON key in `models.json`.
   - `arc` is the accuracy on the [ARC challenge](https://allenai.org/data/arc) dataset.
   - `hellaswag` is the accuracy on the [HellaSwag](https://allenai.org/data/hellaswag) dataset.
   - `truthfulqa` is the accuracy on the [TruthfulQA](https://github.com/sylinrl/TruthfulQA) MC2 dataset.
   - We obtain these metrics using lm-evaluation-harness. See [here](https://github.com/ml-energy/leaderboard/tree/master/pegasus#nlp-benchmark) for specific instructions.

1. Add benchmarking results in CSV files, e.g. `A100_chat_benchmark.csv`. It should be evident from the name of the CSV files which setting the file corresponds to.
