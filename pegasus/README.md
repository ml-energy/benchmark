# Running benchmarks on multiple GPU nodes with Pegasus

[Pegasus](https://github.com/jaywonchung/pegasus) is an SSH-based multi-node command runner.
Different models have different verbosity, and benchmarking takes vastly different amounts of time.
Therefore, we want an automated piece of software that drains a queue of benchmarking jobs (one job per model) on a set of GPUs.

## Setup

### Install Pegasus

Pegasus needs to keep SSH connections with all the nodes in order to queue up and run jobs over SSH.
So you should install and run Pegasus on a computer that you can keep awake.

If you already have Rust set up:

```console
$ cargo install pegasus-ssh
```

Otherwise, you can set up Rust [here](https://www.rust-lang.org/tools/install), or just download Pegasus release binaries [here](https://github.com/jaywonchung/pegasus/releases/latest).

### Necessary setup for each node

Every node must have two things:

1. This repository cloned under `~/workspace/leaderboard`.
  - If you want a different path, search and replace in `spawn-containers.yaml`.
2. Model weights under `/data/leaderboard/weights`.
  - If you want a different path, search and replace in `setupspawn-containers.yaml` and `benchmark.yaml`.

### Specify node names for Pegasus

Modify `hosts.yaml` with nodes. See the file for an example.

- `hostname`: List the hostnames you would use in order to `ssh` into the node, e.g. `jaywonchung@gpunode01`.
- `gpu`: We want to create one Docker container for each GPU. List the indices of the GPUs you would like to use for the hosts.

### Set up Docker containers on your nodes with Pegasus

This spawns one container per GPU (named `leaderboard%d`), for every node.

```console
$ cd pegasus
$ cp spawn-containers.yaml queue.yaml
$ pegasus b
```

`b` stands for broadcast. Every command is run once on all (`hostname`, `gpu`) combinations.

## System benchmark

This will benchmark each model and get you data for the columns `energy`, `throughput`, `latency`, and `response_length`.

Use Pegasus to run benchmarks for all the models across all nodes.

```console
$ cd pegasus
$ cp benchmark.yaml queue.yaml
$ pegasus q
```

`q` stands for queue. Each command is run once on the next available (`hostname`, `gpu`) combination.

After all the tasks finish, aggregate all the data into one node and run [`compute_system_metrics.py`](../scripts/compute_system_metrics.py) to generate CSV files that the leaderboard can display.

## NLP benchmark

We'll use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/72b7f0c00a6ff94632c5b873fc24e093ae74fa47) to run models through three NLP datasets: ARC challenge (`arc`), HellaSwag (`hellaswag`), and TruthfulQA (`truthfulqa`).

Use Pegasus to run benchmarks for all the models across all nodes.

```console
$ cd pegasus
$ cp nlp-eval.yaml queue.yaml
$ pegasus q
```

After all the tasks finish, aggregate all the data into one node and run [`aggregate_nlp_metrics.py`](../scripts/aggregate_nlp_metrics.py) to generate a single `score.csv` that the leaderboard can display.

### Dealing with OOM

Some tasks might run out of memory, in which case you should create a container with more GPUs:

1. Create a container with two GPUs, for example:

```console
$ docker run -dit \
    --name leaderboard01 \
    --gpus '"device=0,1"' \
    -v /data/leaderboard:/data/leaderboard \
    -v $HOME/workspace/leaderboard:/workspace/leaderboard \
    mlenergy/leaderboard:latest bash
```

2. Revise `nlp-eval.yaml` and run with Pegasus, or run directly like this on LLaMA 7B and ARC, for example:

```console
$ docker exec leaderboard01 \
    python lm-evaluation-harness/main.py \
    --device cuda \
    --no_cache \
    --model hf-causal-experimental \
    --model_args pretrained=/data/leaderboard/weights/metaai/llama-7B,trust_remote_code=True,use_accelerate=True \
    --tasks arc_challenge \
    --num_fewshot 25
```
