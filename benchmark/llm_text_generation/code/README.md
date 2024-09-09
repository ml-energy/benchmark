# LLM Text Generation (Code)

This benchmark suite benchmarks vLLM and TGI with the code generation task.

## Setup

### Docker images

You can pull vLLM and TGI Docker images with:

```sh
docker pull mlenergy/vllm:v0.5.4-openai
docker pull mlenergy/tgi:v2.0.2
```

### Installing Benchmark Script Dependencies

```sh
pip install -r requirements.txt
```

### Starting the NVML container

Changing the power limit requires the `SYS_ADMIN` Linux security capability, which we delegate to a daemon Docker container running a base CUDA image.

```sh
bash ../../common/start_nvml_container.sh
```

With the `nvml` container running, you can change power limit with something like `docker exec nvml nvidia-smi -i 0 -pl 200`.

### HuggingFace cache directory

The scripts assume the HuggingFace cache directory will be under `/data/leaderboard/hfcache` on the node that runs this benchmark.


## Benchmarking

### Obtaining one datapoint

Export your HuggingFace hub token as environment variable `$HF_TOKEN`.

The script `scripts/benchmark_one_datapoint.py` assumes that it was run from the directory where `scripts` is, like this:
```sh
python scripts/benchmark_one_datapoint.py --help
```

### Obtaining all datapoints for a single model

Run `scripts/benchmark_one_model.py`.

### Running the entire suite with Pegasus

You can use [`pegasus`](https://github.com/jaywonchung/pegasus) to run the entire benchmark suite.
Queue and host files are in [`./pegasus`](./pegasus).
