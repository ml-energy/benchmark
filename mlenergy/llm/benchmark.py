"""LLM benchmark runner.

Inspired by https://github.com/vllm-project/vllm/blob/8188196a1c/vllm/benchmarks/serve.py
"""

from __future__ import annotations

import atexit
import asyncio
import gc
import io
import sys
import traceback
import json
import logging
import random
import time
import warnings
import os
import subprocess
from contextlib import redirect_stdout
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar
from pathlib import Path

import tyro
import requests
import numpy as np
import aiohttp
from pydantic import BaseModel
from tqdm.asyncio import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from zeus.show_env import show_env
from zeus.monitor import ZeusMonitor

from mlenergy.llm.datasets import SampleRequest
from mlenergy.llm.workloads import (
    WorkloadConfig,
    ImageChat,
    VideoChat,
    AudioChat,
    OmniChat,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

logger = logging.getLogger("mlenergy.llm.run")


WorkloadT = TypeVar("WorkloadT", bound=WorkloadConfig)


class Args(BaseModel, Generic[WorkloadT]):
    """Data model for benchmark arguments.

    Attributes:
        workload: Workload configuration for the benchmark.
        endpoint_type: Type of API endpoint.
        endpoint: API endpoint path.
        max_concurrency: Maximum number of concurrent requests. When used together
            with `request_rate`, this may reduce the actual request rate if the
            server is not processing requests fast enough to keep up.
        request_rate: Number of requests per second. If this is set to `inf`,
            all requests will be sent at time 0.
        burstiness: Burstiness factor of the request generation. No effect when
            `request_rate` is set to `inf`. Default value is 1, which follows
            a Poisson process. Otherwise, the request intervals follow a gamma
            distribution. A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests. A higher burstiness value (burstiness > 1)
            results in a more uniform arrival of requests.
        ignore_eos: Whether to ignore the end-of-sequence token in the requests.
            Setting this will generate tokens until it reaches the maximum number
            of output tokens specified in the request.
        set_max_tokens: Whether to set the maximum number of output tokens in the
            requests. If set to `True`, the maximum number of output tokens will
            be capped to `SampleRequest.expected_output_len`.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability for sampling.
        temperature: Temperature sampling parameter. Greedy decoding if set to 0.0.
        max_num_seqs: Maximum number of seuqences config to start vLLM with.
        max_num_batched_tokens: Maximum number of batched tokens config to start
            vLLM with. TODO: Investigate its impact on the benchmark.
        percentile_metrics: Comma-separated list of selected metrics to report
            percentiles. This argument specifies the metrics to report percentiles.
            Allowed metric names are "ttft", "tpot", "itl", and "e2el". E2EL means
            the client-side end-to-end latency, which is the time from sending
            the request to receiving the last token of the response.
        metric_percentiles: Comma-separated list of percentiles for selected metrics.
            Default is "50,90,95,99" which reports the 50-th, 90-th, 95-th, and
            99-th percentiles. Use `--percentile-metrics` to select metrics.
        overwrite_results: Whether to overwrite the existing results file. If this
            is not set, the script will immediately exit if the results file exists.
        profile: Whether to enable the torch profiler. vLLM should be launched with
            `VLLM_TORCH_PROFILER_DIR` to enable the profiler.
    """

    # Workload configuration
    workload: WorkloadT
    endpoint_type: Literal["openai", "openai-chat"] = "openai-chat"
    endpoint: str = "/v1/completions"
    max_concurrency: int | None = None
    request_rate: float = float("inf")
    burstiness: float = 1.0
    ignore_eos: bool = False
    set_max_tokens: bool = False
    top_p: float | None = 0.95
    top_k: int | None = None
    min_p: float | None = None
    temperature: float | None = 0.8

    # Server configuration
    server_image: str = "vllm/vllm-openai:v0.10.0"
    max_num_seqs: int
    max_num_batched_tokens: int | None = None

    # Results configuration
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,90,95,99"
    overwrite_results: bool = False

    # Miscellaneous
    profile: bool = False


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


@dataclass
class RequestFuncInput:
    """The input for the request function."""

    prompt: str
    api_url: str
    prompt_len: int
    output_len: int | None
    model: str
    model_name: str | None = None
    logprobs: int | None = None
    extra_body: dict | None = None
    multimodal_contents: list[dict] | None = None
    ignore_eos: bool = False
    language: str | None = None


@dataclass
class RequestFuncOutput:
    """The output of the request function including metrics."""

    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # Inter-token latencies
    tpot: float = 0.0  # Avg next-token latencies
    prompt_len: int = 0
    error: str = ""


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """The async request function for the OpenAI Completions API.

    Args:
        request_func_input: The input for the request function.
        pbar: The progress bar to display the progress.

    Returns:
        The output of the request function.
    """
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    if request_func_input.multimodal_contents:
        raise ValueError(
            "OpenAI Completions API does not support multimodal contents. "
            "Use OpenAI Chat Completions API instead."
        )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.output_len is not None:
            payload["max_tokens"] = request_func_input.output_len
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if chunk_bytes.startswith(":"):
                            continue

                        chunk = chunk_bytes.removeprefix("data: ")

                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!"
                        )
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "profile")), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multimodal_contents:
            content.extend(request_func_input.multimodal_contents)
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.output_len is not None:
            payload["max_completion_tokens"] = request_func_input.output_len
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if chunk_bytes.startswith(":"):
                            continue

                        chunk = chunk_bytes.removeprefix("data: ")

                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
}


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    """
    Asynchronously generates requests at a specified rate with optional burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a SampleRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )

    total_requests = len(input_requests)
    assert total_requests > 0, "No requests provided."

    # Precompute delays among requests to minimize request send laggings
    request_rates = []
    delay_ts = []
    for request_index, request in enumerate(input_requests):
        current_request_rate = request_rate
        request_rates.append(current_request_rate)
        if current_request_rate == float("inf"):
            delay_ts.append(0)
        else:
            theta = 1.0 / (current_request_rate * burstiness)

            # Sample the request interval from the gamma distribution.
            # If burstiness is 1, it follows exponential distribution.
            delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

    # Calculate the cumulative delay time from the first sent out requests.
    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]
    if delay_ts[-1] != 0:
        # All requests should be sent in target_total_delay_s. The following
        # logic would re-scale delay time to ensure the final delay_ts
        # align with target_total_delay_s.
        #
        # NOTE: If we simply accumulate the random delta values
        # from the gamma distribution, their sum would have 1-2% gap
        # from target_total_delay_s. The purpose of the following logic is to
        # close the gap for stablizing the throughput data
        # from different random seeds.
        target_total_delay_s = total_requests / request_rate
        normalize_factor = target_total_delay_s / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]

    start_ts = time.time()
    request_index = 0
    for request_index, request in enumerate(input_requests):
        current_ts = time.time()
        sleep_interval_s = start_ts + delay_ts[request_index] - current_ts
        if sleep_interval_s > 0:
            await asyncio.sleep(sleep_interval_s)
        yield request


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
) -> tuple[BenchmarkMetrics, list[int]]:
    """Calculate the metrics for the benchmark.

    Args:
        input_requests: The input requests.
        outputs: The outputs of the requests.
        dur_s: The duration of the benchmark.
        tokenizer: The tokenizer to use.
        selected_percentiles: The percentiles to select.

    Returns:
        A tuple of the benchmark metrics and the actual output lengths.
    """
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    # ttfts is empty if streaming is not supported by the endpoint
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=(np.mean(ttfts or 0) * 1000).item(),
        std_ttft_ms=(np.std(ttfts or 0) * 1000).item(),
        median_ttft_ms=(np.median(ttfts or 0) * 1000).item(),
        percentiles_ttft_ms=[
            (p, (np.percentile(ttfts or 0, p) * 1000).item())
            for p in selected_percentiles
        ],
        mean_tpot_ms=(np.mean(tpots or 0) * 1000).item(),
        std_tpot_ms=(np.std(tpots or 0) * 1000).item(),
        median_tpot_ms=(np.median(tpots or 0) * 1000).item(),
        percentiles_tpot_ms=[
            (p, (np.percentile(tpots or 0, p) * 1000).item())
            for p in selected_percentiles
        ],
        mean_itl_ms=(np.mean(itls or 0) * 1000).item(),
        std_itl_ms=(np.std(itls or 0) * 1000).item(),
        median_itl_ms=(np.median(itls or 0) * 1000).item(),
        percentiles_itl_ms=[
            (p, (np.percentile(itls or 0, p) * 1000).item())
            for p in selected_percentiles
        ],
        mean_e2el_ms=(np.mean(e2els or 0) * 1000).item(),
        std_e2el_ms=(np.std(e2els or 0) * 1000).item(),
        median_e2el_ms=(np.median(e2els or 0) * 1000).item(),
        percentiles_e2el_ms=[
            (p, (np.percentile(e2els or 0, p) * 1000).item())
            for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


async def benchmark(
    zeus_monitor: ZeusMonitor,
    max_num_seqs: int,
    workload: WorkloadConfig,
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_id: str,
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    set_max_tokens: bool,
    max_concurrency: int | None,
    extra_body: dict | None,
):
    if endpoint_type in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[endpoint_type]
    else:
        raise ValueError(f"Unknown endpoint_type: {endpoint_type}")

    logger.info("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].expected_output_len,
        input_requests[0].multimodal_contents,
    )

    assert test_mm_content is None or isinstance(test_mm_content, dict)
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len if set_max_tokens else None,
        multimodal_contents=test_mm_content,
        ignore_eos=ignore_eos,
        extra_body=extra_body,
    )

    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        logger.info("Initial test run completed. Starting main benchmark run...")

    if profile:
        logger.info("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len if set_max_tokens else None,
            multimodal_contents=test_mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            logger.info("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    logger.info("Traffic request rate: %f req/s", request_rate)

    logger.info("Burstiness factor: %f (%s)", burstiness, distribution)
    if max_concurrency is not None:
        logger.info("Maximum request concurrency: %d", max_concurrency)

    pbar = tqdm(total=len(input_requests))

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    tasks: list[asyncio.Task] = []
    benchmark_start_time = time.perf_counter()
    zeus_monitor.begin_window("entire_benchmark", sync_execution=False)
    zeus_monitor.begin_window("steady_state", sync_execution=False)

    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, mm_content = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multimodal_contents,
        )
        req_model_id = model_id

        request_func_input = RequestFuncInput(
            model=req_model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len if set_max_tokens else None,
            multimodal_contents=mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )

    # Steady state energy measurement
    running_requests = len(tasks)
    assert running_requests == workload.num_requests
    steady_state_mes = None
    for coro in asyncio.as_completed(tasks):
        _ = await coro
        running_requests -= 1
        if running_requests < max_num_seqs:
            # Now, the server's batch size is always less than max_num_seqs.
            # This is the end of the steady state.
            steady_state_mes = zeus_monitor.end_window(
                "steady_state", sync_execution=False
            )
            break
    assert steady_state_mes is not None, "End of steady state not reached."

    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        logger.info("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len if set_max_tokens else None,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            logger.info("Profiler stopped")

    if pbar is not None:
        pbar.close()

    entire_mes = zeus_monitor.end_window("entire_benchmark", sync_execution=False)
    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=workload.tokenizer,
        selected_percentiles=selected_percentiles,
    )

    steady_state_time = steady_state_mes.time
    steady_state_energy = sum(steady_state_mes.gpu_energy.values())

    logger.info("[Benchmark results]")
    logger.info("%-40s: %d", "Total requests", workload.num_requests)
    logger.info("%-40s: %d", "Successful requests", metrics.completed)
    logger.info("%-40s: %d", "Steady state duration (s)", steady_state_time)
    logger.info("%-40s: %.2f", "Steady state energy (J)", steady_state_energy)
    logger.info("%-40s: %.2f", "Benchmark duration (s)", benchmark_duration)
    logger.info("%-40s: %d", "Total input tokens", metrics.total_input)
    logger.info("%-40s: %d", "Total generated tokens", metrics.total_output)
    logger.info("%-40s: %.2f", "Request throughput (req/s)", metrics.request_throughput)
    logger.info(
        "%-40s: %.2f", "Output token throughput (tok/s)", metrics.output_throughput
    )
    logger.info(
        "%-40s: %.2f", "Total Token throughput (tok/s)", metrics.total_token_throughput
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "steady_state_duration": steady_state_time,
        "steady_state_energy": steady_state_energy,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "steady_state_measurement": asdict(steady_state_mes),
        "entire_benchmark_measurement": asdict(entire_mes),
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        logger.info("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        logger.info(
            "%-40s: %-10.2f",
            f"Mean {metric_name} (ms)",
            getattr(metrics, f"mean_{metric_attribute_name}_ms"),
        )
        logger.info(
            "%-40s: %-10.2f",
            f"Median {metric_name} (ms)",
            getattr(metrics, f"median_{metric_attribute_name}_ms"),
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            logger.info("%-40s: %-10.2f", f"P{p_word} {metric_name} (ms)", value)
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    logger.info("=" * 50)

    return result


def spawn_vllm_and_ensure_healthy(
    server_image: str,
    port: int,
    model_id: str,
    hf_token: str,
    hf_home: str,
    gpu_ids: list[int],
    max_num_seqs: int,
    max_num_batched_tokens: int | None,
    log_level: str,
    server_log_filepath: Path,
) -> str:
    """Spawn vLLM server and ensure it is healthy.

    Retuens:
        The container name of the spawned vLLM server.
    """
    gpu_str = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    gpu_str = f'"device={gpu_str}"'
    container_name = f"benchmark-vllm-{''.join(str(gpu_id) for gpu_id in gpu_ids)}"

    assert Path(hf_home).exists(), f"Hugging Face home directory not found: {hf_home}"

    # fmt: off
    server_cmd = [
        "docker", "run",
        "--gpus", gpu_str,
        "--ipc", "host",
        "--net", "host",
        "--name", container_name,
        "-e", f"HF_TOKEN={hf_token}",
        "-e", f"LOG_LEVEL={log_level}",
        "-v", f"{hf_home}:/root/.cache/huggingface",
        server_image,
        "--port", str(port),
        "--model", model_id,
        "--tensor-parallel-size", str(len(gpu_ids)),
        "--gpu-memory-utilization", "0.95",
        "--trust-remote-code",
        "--max-num-seqs", str(max_num_seqs),
        # "--max-num-batched-tokens", str(max_num_batched_tokens),
    ]
    # fmt: on

    logger.info("Spawning vLLM server with command: %s", " ".join(server_cmd))
    logger.info("vLLM container name: %s", container_name)
    logger.info("vLLM logs will be written to %s", server_log_filepath)
    server_log_file = open(server_log_filepath, "w")
    subprocess.Popen(server_cmd, stdout=server_log_file, stderr=server_log_file)

    def kill_server():
        """Kill the vLLM server."""
        logger.info("Killing vLLM server container %s", container_name)
        subprocess.run(["docker", "kill", container_name])
        time.sleep(5)
        subprocess.run(["docker", "rm", container_name])
        logger.info("vLLM server container %s killed and removed", container_name)

    atexit.register(kill_server)

    # Wait until the /health endpoint returns 200 OK
    health_url = f"http://127.0.0.1:{port}/health"
    logger.info("Waiting for vLLM server to become healthy at %s", health_url)
    while True:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server is healthy.")
                break
        except requests.RequestException as e:
            logger.warning("Waiting for vLLM server to become healthy: %s", e)
        time.sleep(1)

    return container_name


def main(args: Args) -> None:
    logger.info("%s", args)

    assert isinstance(args.workload, WorkloadConfig)

    # Exit if the result file exists so that the script is idempotent.
    result_file = args.workload.to_path(of="results")
    if result_file.exists() and not args.overwrite_results:
        logger.info(
            "Result file %s already exists. Exiting immediately. "
            "Specify --overwrite-results to run the benchmark and overwrite results.",
            result_file,
        )
        return

    # Necessary envs
    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    hf_token = os.environ["HF_TOKEN"]
    hf_home = os.environ["HF_HOME"]

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        show_env()
    logger.info("Zeus environment information:\n%s", buffer.getvalue())

    zeus_monitor = ZeusMonitor()

    model_id = args.workload.model_id
    random.seed(args.workload.seed)
    np.random.seed(args.workload.seed)

    port = 8000 + int(cuda_visible_devices.split(",")[0])
    api_url = f"http://127.0.0.1:{port}{args.endpoint}"
    base_url = f"http://127.0.0.1:{port}"

    # Load the dataset.
    input_requests = args.workload.load_requests()

    # Collect the sampling parameters.
    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
        }.items()
        if v is not None
    }

    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0  # Default to greedy decoding.

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()

    spawn_vllm_and_ensure_healthy(
        server_image=args.server_image,
        port=port,
        model_id=model_id,
        hf_token=hf_token,
        hf_home=hf_home,
        gpu_ids=[int(gpu_id) for gpu_id in cuda_visible_devices.split(",")],
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        log_level="INFO",
        server_log_filepath=args.workload.to_path(of="server_log"),
    )

    benchmark_result = asyncio.run(
        benchmark(
            zeus_monitor=zeus_monitor,
            max_num_seqs=args.max_num_seqs,
            workload=args.workload,
            endpoint_type=args.endpoint_type,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            set_max_tokens=args.set_max_tokens,
            max_concurrency=args.max_concurrency,
            extra_body=sampling_params,
        )
    )

    # Save config and results to json
    result_json: dict[str, Any] = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["endpoint_type"] = args.endpoint_type
    result_json["model_id"] = model_id
    result_json["num_prompts"] = args.workload.num_requests

    # Traffic
    result_json["request_rate"] = (
        args.request_rate if args.request_rate < float("inf") else "inf"
    )
    result_json["burstiness"] = args.burstiness
    result_json["max_concurrency"] = args.max_concurrency

    # Merge with benchmark result
    result_json = {**result_json, **benchmark_result}

    # Save to results file
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)


if __name__ == "__main__":
    args = tyro.cli(Args[ImageChat | VideoChat | AudioChat | OmniChat])

    # Set up the logger so that it logs to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s: %(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.workload.to_path(of="driver_log"), mode="w"),
        ],
    )

    try:
        main(args)
    except Exception as e:
        logger.exception("An error occurred during the benchmark: %s", e)
        raise
