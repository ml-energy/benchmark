# ML.ENERGY Leaderboard

How much energy do LLMs consume?

This README focuses on explaining how to run the benchmark yourself.
The actual leaderboard is here: https://ml.energy/leaderboard.

## Setup

### Model weights

- For models that are directly accessible in Hugging Face Hub, you don't need to do anything.
- For other models, convert them to Hugging Face format and put them in `/data/leaderboard/weights/lmsys/vicuna-13B`, for example. The last two path components (e.g., `lmsys/vicuna-13B`) are taken as the name of the model.

### Docker container

```console
$ git clone https://github.com/ml-energy/leaderboard.git
$ cd leaderboard
$ docker build -t ml-energy:latest .
# Replace /data/leaderboard with your data directory.
$ docker run -it \
    --name leaderboard \
    --gpus all \
    -v /data/leaderboard:/data/leaderboard \
    -v $HOME/workspace/leaderboard:/workspace/leaderboard \
    ml-energy:latest bash
```

## Running the benchmark

```console
# Inside the container
$ cd /workspace/leaderboard
$ python scripts/benchmark.py --model-path /data/leaderboard/weights/lmsys/vicuna-13B --input-file sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json
$ python scripts/benchmark.py --model-path databricks/dolly-v2-12b --input-file sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json
```
