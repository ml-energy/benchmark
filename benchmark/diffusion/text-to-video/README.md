# Diffusion model (Text to Video)

This benchmark suite benchmarks diffusion models with the text-to-video task.

## Setup

### Docker images

```sh
docker build -t mlenergy/leaderboard:diffusion-t2v .
```

### HuggingFace cache directory

The scripts assume the HuggingFace cache directory will be under `/data/leaderboard/hfcache` on the node that runs this benchmark.

## Benchmarking

### Obtaining one datapoint

The Docker image we've build runs `python scripts/benchmark_one_datapoint.py` as its `ENTRYPOINT`.

```sh
docker run \
  --gpus '"device=0"' \
  --cap-add SYS_ADMIN \
  -v /data/leaderboard/hfcache:/root/.cache/huggingface 
  -v $(pwd):/workspace/text-to-image \
  mlenergy/leaderboard:diffusion-t2v \
  --result-root results \
  --batch-size 2 \
  --power-limit 300 \
  --save-every 5 \
  --model guoyww/animatediff-motion-adapter-v1-5-3 \
  --dataset-path sharegpt4video/sharegpt4video_100.json \
  --huggingface-token $HF_TOKEN
```

### Obtaining all datapoints for a single model

Export your HuggingFace hub token as environment variable `$HF_TOKEN`.

Run `scripts/benchmark_one_model.py`.

### Running the entire suite with Pegasus

You can use [`pegasus`](https://github.com/jaywonchung/pegasus) to run the entire benchmark suite.
Queue and host files are in [`./pegasus`](./pegasus).
