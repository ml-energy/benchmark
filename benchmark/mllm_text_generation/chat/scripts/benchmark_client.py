from __future__ import annotations

import argparse
import asyncio
import requests
import json
import random
import time
from typing import AsyncGenerator
from dataclasses import asdict, dataclass, field

import pynvml
import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from zeus.monitor import ZeusMonitor


SYSTEM_PROMPT = "You are an artificial intelligence assistant that gives helpful answers to the user's questions or instructions."
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=3 * 3600)


@dataclass
class Results:
    model: str
    backend: str
    gpu_model: str
    num_gpus: int
    max_num_seqs: int
    power_limit: int
    request_rate: float
    num_requests: int
    num_failures: int = 0
    system_prompt: str = SYSTEM_PROMPT
    total_runtime: float = 0.0
    requests_per_second: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    latency_per_request: float = 0.0
    latency_per_output_token: float = 0.0
    server_side_total_energy: float = 0.0
    server_side_energy_per_request: float = 0.0
    server_side_energy_per_output_token: float = 0.0
    server_side_average_power: float = 0.0
    client_side_total_energy: float = 0.0
    client_side_energy_per_request: float = 0.0
    client_side_energy_per_output_token: float = 0.0
    client_side_average_power: float = 0.0
    results: list[Result] = field(default_factory=list)


@dataclass
class ResultIntermediate:
    success: bool = True
    latency: float = 0.0
    prompt: str = ""
    response_bytes: list[bytes] = field(default_factory=list)


@dataclass
class Result:
    success: bool = True
    latency: float = 0.0
    prompt: str = ""
    response: str = ""
    num_prompt_tokens: int = 0
    num_completion_tokens: int = 0
    energy: float = 0.0


def load_llava_instruct(path: str) -> list[tuple[str, str]]:
    # Load the dataset.
    with open(path) as f:
        dataset = json.load(f)

    # Only keep the first turn of each conversation.
    return [(sample["image"], sample["prompt"]) for sample in dataset]


async def get_request(
    input_requests: list[tuple[str, str]],
    result_intermediates: list[ResultIntermediate],
    request_rate: float,
) -> AsyncGenerator[tuple[ResultIntermediate, tuple[str, str]], None]:
    if request_rate == float("inf"):
        # If the request rate is infinity, then we don't need to wait.
        for item in zip(result_intermediates, input_requests, strict=True):
            yield item
        return

    for item in zip(result_intermediates, input_requests, strict=True):
        yield item

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    result_intermediate: ResultIntermediate,
    backend: str,
    model: str,
    api_url: str,
    image: str,
    prompt: str,
) -> None:
    headers = {"Content-Type": "application/json"}
    # OpenAI Chat Completions API request format
    # Assuming `add_generation_prompt` is either not needed or set to true
    pload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",
                    },
                },
            ]},
        ],
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
        "stop": ["\nUser:", "<|endoftext|>", "</s>"],
    }

    async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as session:
        request_start_time = time.perf_counter()
        async with session.post(api_url, headers=headers, json=pload) as response:
            # Request failed
            if response.status >= 300:
                print(f"Request failed: {await response.text()}")
                result_intermediate.prompt = prompt
                result_intermediate.success = False
                return
            chunks = []
            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
            request_end_time = time.perf_counter()

    result_intermediate.latency = request_end_time - request_start_time
    result_intermediate.prompt = prompt
    result_intermediate.response_bytes = chunks


async def benchmark(
    results: Results,
    backend: str,
    model: str,
    api_url: str,
    input_requests: list[tuple[str, str]],
    request_rate: float,
) -> None:
    tasks: list[asyncio.Task] = []
    result_intermediates = [ResultIntermediate() for _ in input_requests]
    pbar = tqdm(total=len(input_requests))
    async for ri, (image, prompt) in get_request(input_requests, result_intermediates, request_rate):
        pbar.update(1)
        task = asyncio.create_task(
            # Ensures results has same ordering as the input dataset
            send_request(ri, backend, model, api_url, image, prompt)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)

    for result, intermediate in zip(results.results, result_intermediates, strict=True):
        result.success = intermediate.success
        result.latency = intermediate.latency
        result.prompt = intermediate.prompt
        if result.success:
            output = json.loads(b"".join(intermediate.response_bytes).decode("utf-8"))
            result.response = output["choices"][0]["message"]["content"]
            result.num_prompt_tokens = output["usage"]["prompt_tokens"]
            result.num_completion_tokens = output["usage"]["completion_tokens"]
            result.energy = output["usage"]["energy"]


def run_benchmark(
    args: argparse.Namespace,
    api_url: str,
    input_requests: list[tuple[str, str]],
    out_filename: str,
):
    zeus_monitor = ZeusMonitor()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_model = pynvml.nvmlDeviceGetName(handle)
    pynvml.nvmlShutdown()

    results = Results(
        model=args.model,
        backend=args.backend,
        gpu_model=gpu_model,
        num_gpus=len(zeus_monitor.gpu_indices),
        max_num_seqs=args.max_num_seqs,
        power_limit=args.power_limit,
        request_rate=args.request_rate,
        num_requests=len(input_requests),
        results=[Result() for _ in input_requests],
    )

    zeus_monitor.begin_window(out_filename, sync_execution=False)
    asyncio.run(benchmark(results, args.backend, args.model, api_url, input_requests, args.request_rate))
    measurements = zeus_monitor.end_window(out_filename, sync_execution=False)

    client_side_total_energy = measurements.total_energy

    # Store aggregated results
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency = 0
    total_latency_per_output_token = 0
    server_side_total_energy = 0
    for result in results.results:
        if not result.success:
            results.num_failures += 1
            continue
        total_prompt_tokens += result.num_prompt_tokens
        total_completion_tokens += result.num_completion_tokens
        total_latency += result.latency
        total_latency_per_output_token += result.latency / result.num_completion_tokens
        server_side_total_energy += result.energy

    num_results = len(results.results) - results.num_failures
    if num_results == 0:
        raise RuntimeError("All requests failed!")

    results.total_runtime = measurements.time
    results.requests_per_second = num_results / results.total_runtime
    results.total_prompt_tokens = total_prompt_tokens
    results.total_completion_tokens = total_completion_tokens
    results.latency_per_request = total_latency / num_results
    results.latency_per_output_token = total_latency_per_output_token / num_results
    results.server_side_total_energy = server_side_total_energy
    results.server_side_energy_per_request = results.server_side_total_energy / num_results
    results.server_side_energy_per_output_token = results.server_side_total_energy / results.total_completion_tokens
    results.server_side_average_power = server_side_total_energy / results.total_runtime
    results.client_side_total_energy = client_side_total_energy
    results.client_side_energy_per_request = client_side_total_energy / num_results
    results.client_side_energy_per_output_token = client_side_total_energy / results.total_completion_tokens
    results.client_side_average_power = client_side_total_energy / results.total_runtime

    with open(out_filename, "w") as f:
        f.write(json.dumps(asdict(results), indent=2))
    print("Benchmark results written to", out_filename)

    print("Benchmark results:")
    print(f"Model: {results.model}")
    print(f"Backend: {results.backend}")
    print(f"Request rate: {results.request_rate} requests/s")
    print()
    print(f"Total benchmark runtime: {results.total_runtime:.2f} s")
    print(f"Total number of requests: {results.num_requests}")
    print(f"Number of failed requests: {results.num_failures}")
    print(f"Requests per second: {results.requests_per_second:.2f} requests/s")
    print(f"Average latency per request: {results.latency_per_request:.2f} s")
    print(f"Average latency per output token: {results.latency_per_output_token:.2f} s")
    print(f"(Client-side) Total energy: {results.client_side_total_energy:.2f} J")
    print(f"(Client-side) Energy per request: {results.client_side_energy_per_request:.2f} J")
    print(f"(Client-side) Energy per token: {results.client_side_energy_per_output_token:.2f} J")
    print(f"(Client-side) Average power: {results.client_side_average_power:.2f} W")
    print(f"(Server-side) Total energy: {results.server_side_total_energy:.2f} J")
    print(f"(Server-side) Energy per request: {results.server_side_energy_per_request:.2f} J")
    print(f"(Server-side) Energy per token: {results.server_side_energy_per_output_token:.2f} J")
    print(f"(Server-side) Average power: {results.server_side_average_power:.2f} W")


def wait_server_ready(list_models_url: str) -> None:
    while True:
        try:
            response = requests.get(list_models_url)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException:
            print("Waiting for the server to be ready...")
            time.sleep(1)


def main(args: argparse.Namespace):
    if args.backend not in ["tgi", "vllm"]:
        raise ValueError(f"Unknown backend: {args.backend}")

    arg_out_filename = f"{args.benchmark_name}+args.json"
    with open(arg_out_filename, "w") as f:
        f.write(json.dumps(vars(args), indent=2))
    print(args)
    print("Benchmark args written to", arg_out_filename)

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://localhost:{args.port}/v1/chat/completions"
    input_requests = load_llava_instruct(args.dataset_path)

    if args.backend == "vllm":
        wait_server_ready(f"http://localhost:{args.port}/v1/models")
    elif args.backend == "tgi":
        wait_server_ready(f"http://localhost:{args.port}/health")
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    run_benchmark(args, api_url, input_requests, f"{args.benchmark_name}+results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["vllm", "tgi"], help="Server to benchmark.")
    parser.add_argument("--port", type=int, required=True, help="Port of the server to benchmark.")
    parser.add_argument("--model", required=True, help="Model to benchmark, e.g., meta-llama/Llama-2-7b-chat-hf.")
    parser.add_argument("--dataset-path", help="Path to the LLava Instruct dataset to feed to the server.")
    parser.add_argument(
        "--request-rate",
        type=float,
        required=True,
        help="Poisson process rate for request arrival times. If this is inf, all requests are sent at time 0.",
    )
    parser.add_argument(
        "--benchmark-name",
        required=True,
        help="Name of the benchmark. Result files will be written to paths derived from this.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--power-limit", type=int, required=True, help="Not used but passed in in order to save to results file.")
    parser.add_argument("--max-num-seqs", type=int, required=True, help="Not used but passed in in order to save to results file.")
    parser.add_argument("--data-dup-factor", type=int, default=1, help="How many times to repeat the dataset to generate more requests.")
    args = parser.parse_args()
    main(args)
