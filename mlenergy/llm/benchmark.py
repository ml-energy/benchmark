"""LLM benchmark runner.

Inspired by https://github.com/vllm-project/vllm/blob/8188196a1c/vllm/benchmarks/serve.py
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import gc
import io
import json
import logging
import os
import random
import resource
import subprocess
import sys
import time
import warnings
import traceback
from collections.abc import AsyncGenerator
from typing import Any, Generic, Literal, TypeVar
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import tyro
import numpy as np
import requests
from pydantic import BaseModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from zeus.monitor import ZeusMonitor, PowerMonitor
from zeus.show_env import show_env

from mlenergy.llm.datasets import SampleRequest
from mlenergy.llm.workloads import (
    AudioChat,
    GPQA,
    ImageChat,
    LMArenaChat,
    OmniChat,
    VideoChat,
    WorkloadConfig,
    LengthControl,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

logger = logging.getLogger("mlenergy.llm.run")


WorkloadT = TypeVar("WorkloadT", bound=WorkloadConfig)


class Args(BaseModel, Generic[WorkloadT]):
    """Data model for benchmark arguments.

    Attributes:
        workload: Workload configuration for the benchmark.
        endpoint_type: Type of API endpoint.
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
            This will lead to all requests generating tokens until they reach
            their maximum output token parameter in the request. Default is `False`.
        max_output_tokens: Maximum number of output tokens to set for each request.
            If set to `"dataset"`, `SampleRequest.expected_output_len` is used.
            If set to an integer, all requests are capped to that number. Finally,
            if set to `None`, output length is not capped at all, and some requests
            might generate forever. Default is `None`.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability for sampling.
        temperature: Temperature sampling parameter. Greedy decoding if set to 0.0.
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
    """

    # Workload configuration
    workload: WorkloadT
    endpoint_type: Literal["openai", "openai-chat"] = "openai-chat"
    max_concurrency: int | None = None
    request_rate: float = float("inf")
    burstiness: float = 1.0
    ignore_eos: bool = False
    max_output_tokens: int | Literal["dataset"] | None = None
    top_p: float | None = 0.95
    top_k: int | None = None
    min_p: float | None = None
    temperature: float | None = 0.8

    # Server configuration
    server_image: str = "vllm/vllm-openai:v0.10.0"

    # Results configuration
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,90,95,99"
    overwrite_results: bool = False


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

    prompt: str | list[str]
    api_url: str
    prompt_len: int
    output_len: int | None
    model: str
    extra_body: dict | None = None
    multimodal_contents: list[dict] | None = None
    ignore_eos: bool = False


@dataclass
class RequestFuncOutput:
    """The output of the request function including metrics."""

    prompt: str | list[str] = ""
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # Inter-token latencies
    tpot: float = 0.0  # Avg next-token latencies
    prompt_len: int = 0
    error: str = ""


class RequestTracker:
    """Tracks request states and generate tokens.

    Start of steady state: When `max_num_seqs` requests have sent back their first
    output tokens, we assume the server has ramped up to its steady state.

    End of steady state: When `num_requests - max_num_seqs - 1` requests have
    completely finished, we assume the server has begun ramping down from its steady
    state. The `-1` is there to end the steady state measurement slightly early
    to make sure we don't include the ramp down phase at all.
    """

    def __init__(self, max_num_seqs: int, num_requests: int, log: bool = True) -> None:
        """Initialize the CounterWaiter with a target value."""
        self.start_event = asyncio.Event()
        self.end_event = asyncio.Event()

        self.max_num_seqs = max_num_seqs
        self.num_requests = num_requests
        self.log = log

        self.num_generated_tokens = 0
        self.num_started = 0
        self.num_finished = 0

    async def wait_start(self) -> None:
        """Wait until the steady state starts."""
        await self.start_event.wait()

    async def wait_end(self) -> None:
        """Wait until the steady state ends."""
        await self.end_event.wait()

    def notify_request_started(self) -> None:
        """Notify that a request has started.

        This should be called when the first token of the request is received.
        """
        self.num_started += 1
        if self.num_started >= self.max_num_seqs:
            self.start_event.set()

    def notify_tokens_generated(self, num_tokens: int) -> None:
        """Notify that a token has been generated."""
        self.num_generated_tokens += num_tokens

    def notify_request_finished(self) -> None:
        """Notify that a request has finished."""
        self.num_finished += 1
        if self.num_finished >= self.num_requests - self.max_num_seqs - 1:
            self.end_event.set()
        if self.log:
            logger.info(
                "%d/%d requests finished with %d cumulative tokens generated across all requests.",
                self.num_finished,
                self.num_requests,
                self.num_generated_tokens,
            )

    def get_num_generated_tokens(self) -> int:
        """Get the number of generated tokens."""
        return self.num_generated_tokens

    def get_num_started(self) -> int:
        """Get the number of started requests."""
        return self.num_started


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    request_tracker: RequestTracker,
) -> RequestFuncOutput:
    """The async request function for the OpenAI Completions API.

    Args:
        request_func_input: The input for the request function.
        request_tracker: Tracks request states and generated tokens.

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

    if not isinstance(request_func_input.prompt, str):
        raise ValueError(
            "OpenAI Completions API only supports single string prompt, "
            "not a list of strings."
        )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }
        # Especially for the Completions API, not setting this will default to 16 tokens.
        # We need to explicitly set it to `None` to disable the limit.
        payload["max_tokens"] = request_func_input.output_len
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {}

        output = RequestFuncOutput()
        output.prompt = request_func_input.prompt
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
                    current_completion_tokens = 0
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
                                    request_tracker.notify_request_started()
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                                if usage := data.get("usage"):
                                    if completion_tokens := usage.get(
                                        "completion_tokens"
                                    ):
                                        if (
                                            completion_tokens
                                            != current_completion_tokens
                                        ):
                                            inc = (
                                                completion_tokens
                                                - current_completion_tokens
                                            )
                                            request_tracker.notify_tokens_generated(inc)
                                            current_completion_tokens = (
                                                completion_tokens
                                            )
                                            for _ in range(inc - 1):
                                                output.itl.append(0)
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
        finally:
            request_tracker.notify_request_finished()

    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    request_tracker: RequestTracker,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "profile")), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    if isinstance(request_func_input.prompt, str):
        content = [{"type": "text", "text": request_func_input.prompt}]
        content.extend(request_func_input.multimodal_contents or [])
        messages = [{"role": "user", "content": content}]
    else:
        content = [{"type": "text", "text": request_func_input.prompt[0]}]
        content.extend(request_func_input.multimodal_contents or [])
        messages = [{"role": "user", "content": content}]
        for i, prompt in enumerate(request_func_input.prompt[1:]):
            role = "user" if i % 2 == 1 else "assistant"
            messages.append(
                {"role": role, "content": [{"type": "text", "text": prompt}]}
            )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model,
            "messages": messages,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }
        payload["max_completion_tokens"] = request_func_input.output_len
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
        }

        output = RequestFuncOutput()
        output.prompt = request_func_input.prompt
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
                    current_completion_tokens = 0
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
                                    request_tracker.notify_request_started()
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""
                                if usage := data.get("usage"):
                                    if completion_tokens := usage.get(
                                        "completion_tokens"
                                    ):
                                        if (
                                            completion_tokens
                                            != current_completion_tokens
                                        ):
                                            inc = (
                                                completion_tokens
                                                - current_completion_tokens
                                            )
                                            request_tracker.notify_tokens_generated(inc)
                                            current_completion_tokens = (
                                                completion_tokens
                                            )
                                            for _ in range(inc - 1):
                                                output.itl.append(0)
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
        finally:
            request_tracker.notify_request_finished()

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
    model_id: str,
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    max_output_tokens: int | Literal["dataset"] | None,
    max_concurrency: int | None,
    extra_body: dict | None,
):
    if endpoint_type in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[endpoint_type]
    else:
        raise ValueError(f"Unknown endpoint_type: {endpoint_type}")

    logger.info("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].multimodal_contents,
    )

    assert test_mm_content is None or isinstance(test_mm_content, list)
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt if isinstance(test_prompt, str) else test_prompt[0],
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=20,
        multimodal_contents=test_mm_content,
        ignore_eos=ignore_eos,
        extra_body=extra_body,
    )

    test_output = await request_func(
        request_func_input=test_input,
        request_tracker=RequestTracker(8, 100, log=False),
    )
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        logger.info("Initial test run completed. Starting warmup...")

    # Warmup
    await asyncio.gather(
        *[
            request_func(
                request_func_input=test_input,
                request_tracker=RequestTracker(8, 100, log=False),
            )
            for _ in range(10)
        ]
    )
    logger.info("Warmup completed. Starting benchmark...")

    # Zeus power monitor
    power_monitor = PowerMonitor(update_period=0.05)

    logger.info("Traffic request rate: %f req/s", request_rate)

    if request_rate == float("inf"):
        logger.info(
            "Request rate is set to infinity, all requests will be sent at once."
        )
    else:
        distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"
        logger.info("Burstiness factor: %f (%s)", burstiness, distribution)

    if max_concurrency is not None:
        logger.info("Maximum request concurrency: %d", max_concurrency)

    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency
        else contextlib.nullcontext()
    )
    request_tracker = RequestTracker(
        max_num_seqs=max_num_seqs, num_requests=workload.num_requests
    )

    async def limited_request_func(request_func_input):
        async with semaphore:
            return await request_func(
                request_func_input=request_func_input, request_tracker=request_tracker
            )

    tasks: list[asyncio.Task] = []
    benchmark_start_time = time.time()
    zeus_monitor.begin_window("entire_benchmark", sync_execution=False)

    async for request in get_request(input_requests, request_rate, burstiness):
        # None -> No cap
        # int -> Use the specified static maximum output length
        # "dataset" -> Use the expected output length from the dataset
        if max_output_tokens is None:
            output_len = None
        elif isinstance(max_output_tokens, int):
            output_len = max_output_tokens
        elif max_output_tokens == "dataset":
            output_len = request.expected_output_len
        else:
            raise ValueError(f"Unexpected max_output_tokens value: {max_output_tokens}")

        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=output_len,
            multimodal_contents=request.multimodal_contents,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input)
            )
        )

    # Steady state energy measurement
    # Let's say the server's maximum batch size is B and we send a total of N requests.
    # After B requests send back their first output tokens, we can expect the server to
    # have ramped up to its steady state after the initial prefill burst. Then, when
    # N - B requests have sent back their first output tokens, it means that the current
    # queue length is B, and soon when one more request is completed, the server will
    # exit the steady state and enter the ramp down phase. Thus, for our steady state
    # measurement, we slice the time range between the moment B requests have sent back
    # their first token and the moment N - B requests have sent back their first token.
    await request_tracker.wait_start()
    steady_state_start_time = time.time()
    steady_state_token_begin = request_tracker.get_num_generated_tokens()
    steady_state_num_started_begin = request_tracker.get_num_started()
    zeus_monitor.begin_window("steady_state", sync_execution=False)
    logger.info("Steady state has begun.")
    await request_tracker.wait_end()
    steady_state_end_time = time.time()
    steady_state_token_end = request_tracker.get_num_generated_tokens()
    steady_state_num_started_end = request_tracker.get_num_started()
    steady_state_mes = zeus_monitor.end_window("steady_state", sync_execution=False)
    logger.info("Steady state finished.")
    steady_state_tokens = steady_state_token_end - steady_state_token_begin
    steady_state_num_requests = (
        steady_state_num_started_end - steady_state_num_started_begin
    )

    # Gather the rest of the requests.
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    entire_mes = zeus_monitor.end_window("entire_benchmark", sync_execution=False)
    benchmark_end_time = time.time()
    benchmark_duration = benchmark_end_time - benchmark_start_time

    power_timeline = power_monitor.get_all_power_timelines(
        start_time=benchmark_start_time,
        end_time=benchmark_end_time,
    )

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=workload.tokenizer,
        selected_percentiles=selected_percentiles,
    )

    steady_state_time = steady_state_mes.time
    steady_state_energy = sum(steady_state_mes.gpu_energy.values())
    steady_state_prefill_energy = None
    steady_state_prefill_energy_per_token = None
    steady_state_decode_energy = None
    steady_state_decode_energy_per_token = None
    if workload.num_prefills and workload.num_decodes:
        # separate prefill and decode energy
        steady_state_prefill_energy = sum(
            [steady_state_mes.gpu_energy[p] for p in range(workload.num_prefills)]
        )
        # The number of requests that started (i.e., got their first token) during the
        # steady state gives us the number of prefills done.
        steady_state_prefill_energy_per_token = (
            steady_state_prefill_energy / steady_state_num_requests
        )
        steady_state_decode_energy = sum(
            [
                steady_state_mes.gpu_energy[d]
                for d in range(
                    workload.num_prefills,
                    workload.num_prefills + workload.num_decodes,
                )
            ]
        )
        steady_state_decode_energy_per_token = steady_state_decode_energy / (
            steady_state_tokens - steady_state_num_requests
        )

    steady_state_energy_per_token = steady_state_energy / steady_state_tokens
    entire_benchmark_energy = sum(entire_mes.gpu_energy.values())
    entire_benchmark_energy_per_token = (
        entire_benchmark_energy / request_tracker.get_num_generated_tokens()
    )

    # here we use the steady state energy per token to calculate per generation energy
    energy_per_generation = [
        steady_state_energy_per_token * output_len for output_len in actual_output_lens
    ]

    logger.info("{s:{c}^{n}}".format(s="Benchmark results", n=51, c="="))
    logger.info("%-40s: %d", "Total requests", workload.num_requests)
    logger.info("%-40s: %d", "Successful requests", metrics.completed)
    logger.info("%-40s: %.2f", "Benchmark total duration (s)", benchmark_duration)
    logger.info("%-40s: %.2f", "Benchmark total energy (J)", entire_benchmark_energy)
    logger.info(
        "%-40s: %.2f",
        "Benchmark total energy (J) per token",
        entire_benchmark_energy_per_token,
    )
    logger.info("%-40s: %d", "Steady state duration (s)", steady_state_time)
    logger.info("%-40s: %.2f", "Steady state energy (J)", steady_state_energy)
    if steady_state_prefill_energy is not None:
        logger.info(
            "%-40s: %.2f",
            "Steady state prefill energy (J)",
            steady_state_prefill_energy,
        )
        logger.info(
            "%-40s: %.2f",
            "Steady state prefill energy (J) per token",
            steady_state_prefill_energy_per_token,
        )
    if steady_state_decode_energy is not None:
        logger.info(
            "%-40s: %.2f",
            "Steady state decode energy (J)",
            steady_state_decode_energy,
        )
        logger.info(
            "%-40s: %.2f",
            "Steady state decode energy (J) per token",
            steady_state_decode_energy_per_token,
        )
    logger.info(
        "%-40s: %.2f",
        "Steady state energy (J) per token",
        steady_state_energy_per_token,
    )
    logger.info("%-40s: %d", "Total input tokens", metrics.total_input)
    logger.info(
        "%-40s: %d", "Total generated tokens (usage stats)", metrics.total_output
    )
    logger.info(
        "%-40s: %d",
        "Total generated tokens (counted)",
        request_tracker.get_num_generated_tokens(),
    )
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
        "steady_state_prefill_energy": steady_state_prefill_energy,
        "steady_state_prefill_energy_per_token": steady_state_prefill_energy_per_token,
        "steady_state_decode_energy": steady_state_decode_energy,
        "steady_state_decode_energy_per_token": steady_state_decode_energy_per_token,
        "steady_state_energy_per_token": steady_state_energy_per_token,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "steady_state_measurement": asdict(steady_state_mes),
        "entire_benchmark_measurement": asdict(entire_mes),
        "results": [
            {
                "prompt": output.prompt,
                "generated_text": output.generated_text,
                "input_len": output.prompt_len,
                "output_len": output_len,
                "latency": output.latency,
                "ttft": output.ttft,
                "itl": output.itl,
                "error": output.error,
                "energy": energy,
            }
            for output, output_len, energy in zip(
                outputs,
                actual_output_lens,
                energy_per_generation,
                strict=True,
            )
        ],
        "power_timeline": {
            "benchmark_start_time": benchmark_start_time,
            "benchmark_end_time": benchmark_end_time,
            "steady_state_start_time": steady_state_start_time,
            "steady_state_end_time": steady_state_end_time,
            "power": power_timeline,
        },
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
        logger.info("{s:{c}^{n}}".format(s=metric_header, n=51, c="-"))
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


def spawn_vllm(
    server_image: str,
    port: int,
    discovery_port: int,
    model_id: str,
    hf_token: str,
    hf_home: str,
    gpu_ids: list[int],
    max_num_seqs: int,
    max_num_batched_tokens: int | None,
    log_level: str,
    server_log_filepath: Path,
    vllm_cache_dir: str | None = None,
    num_prefills: int = 0,
    num_decodes: int = 0,
) -> list[str]:
    """Spawn vLLM server(s).

    This does not wait for the server to be ready.

    Retuens:
        The container names of the spawned vLLM servers.
    """
    assert Path(hf_home).exists(), f"Hugging Face home directory not found: {hf_home}"

    spawned_containers = []
    spawned_processes = []
    if num_prefills != 0 and num_decodes != 0:
        # PD
        if len(gpu_ids) < num_decodes + num_prefills:
            raise ValueError(
                "Number of GPUs must be greater than or equal to "
                "num_prefills + num_decodes."
            )
        # start proxy server first at port
        proxy_filepath = server_log_filepath.with_name("proxy_server_log.txt")
        proxy_log_file = open(proxy_filepath, "w")
        proxy_server_cmd = [
            "python3",
            "-u",
            "scripts/disagg_proxy_p2p_nccl_xpyd.py",
            "--port",
            str(port),
            "--discovery-port",
            str(discovery_port),
            "--num-prefills",
            str(num_prefills),
            "--num-decodes",
            str(num_decodes),
        ]
        logger.info(
            "Spawning vLLM proxy server with command: %s",
            " ".join(proxy_server_cmd),
        )
        proxy_handle = subprocess.Popen(
            proxy_server_cmd,
            stdout=proxy_log_file,
            stderr=proxy_log_file,
        )
        spawned_processes.append(proxy_handle)
        gpu_per_instance = len(gpu_ids) // (num_prefills + num_decodes)
        remaining_gpus = len(gpu_ids) % (num_prefills + num_decodes)
        if remaining_gpus != 0:
            raise ValueError(
                "Number of GPUs must be divisible by num_prefills + num_decodes."
            )

        prefill_http_base_port = port + 1000
        prefill_kv_base_port = port + 2000
        for i in range(num_prefills):
            cur_gpus = gpu_ids[i * gpu_per_instance : (i + 1) * gpu_per_instance]
            gpu_str = ",".join(str(gpu_id) for gpu_id in cur_gpus)
            container_name = f"benchmark-vllm-prefill-{i}-{''.join(str(gpu_id) for gpu_id in cur_gpus)}"
            # fmt: off
            prefill_kv_transfer_config = {
                "kv_connector": "P2pNcclConnector",
                "kv_role": "kv_producer",
                "kv_buffer_size": "1e1",
                "kv_port": str(prefill_kv_base_port + i),
                "kv_connector_extra_config": {
                    "proxy_ip": "0.0.0.0",
                    "proxy_port": str(discovery_port),
                    "http_port": str(prefill_http_base_port + i),
                    "send_type": "PUT_ASYNC",
                    "nccl_num_channels": "16",
                }
            }
            server_cmd = [
                "docker", "run",
                # nccl needs peer GPUs to be visible,
                # so we give all GPUs to the container
                # while limiting the visible GPUs to vLLM
                "--gpus", "all",
                "-e", "CUDA_VISIBLE_DEVICES=" + gpu_str,
                "--ipc", "host",
                "--pid", "host",
                "--net", "host",
                "--name", container_name,
                "-e", f"HF_TOKEN={hf_token}",
                "-e", f"LOG_LEVEL={log_level}",
                "-v", f"{hf_home}:/root/.cache/huggingface",
                *(
                    ["-v", f"{vllm_cache_dir}:/root/.cache/vllm/torch_compile_cache"] if vllm_cache_dir else []
                ),
                server_image,
                "--port", str(prefill_http_base_port + i),
                "--model", model_id,
                "--tensor-parallel-size", str(len(cur_gpus)),
                "--gpu-memory-utilization", "0.9",
                "--trust-remote-code",
                "--max-num-seqs", str(max_num_seqs),
                "--kv-transfer-config", json.dumps(prefill_kv_transfer_config),
            ]
            # fmt: on
            if max_num_batched_tokens is not None:
                server_cmd.extend(
                    ["--max-num-batched-tokens", str(max_num_batched_tokens)]
                )

            logger.info("Spawning vLLM server with command: %s", " ".join(server_cmd))
            logger.info("vLLM container name: %s", container_name)
            prefill_log_filepath = server_log_filepath.with_name(
                f"prefill_{i}_server_log.txt"
            )
            logger.info("vLLM logs will be written to %s", prefill_log_filepath)
            prefill_server_log_file = open(prefill_log_filepath, "w")
            subprocess.Popen(
                server_cmd,
                stdout=prefill_server_log_file,
                stderr=prefill_server_log_file,
            )
            spawned_containers.append(container_name)

        decode_http_base_port = port + 3000
        decode_kv_base_port = port + 4000
        decode_gpu_ids = gpu_ids[num_prefills * gpu_per_instance :]
        for i in range(num_decodes):
            cur_gpus = decode_gpu_ids[i * gpu_per_instance : (i + 1) * gpu_per_instance]
            gpu_str = ",".join(str(gpu_id) for gpu_id in cur_gpus)
            container_name = f"benchmark-vllm-decode-{i}-{''.join(str(gpu_id) for gpu_id in cur_gpus)}"
            # fmt: off
            decode_kv_transfer_config = {
                "kv_connector": "P2pNcclConnector",
                "kv_role": "kv_consumer",
                "kv_buffer_size": "8e9",
                "kv_port": str(decode_kv_base_port + i),
                "kv_connector_extra_config": {
                    "proxy_ip": "0.0.0.0",
                    "proxy_port": str(discovery_port),
                    "http_port": str(decode_http_base_port + i),
                    "send_type": "PUT_ASYNC",
                    "nccl_num_channels": "16",
                }
            }
            server_cmd = [
                "docker", "run",
                # nccl needs peer GPUs to be visible,
                # so we give all GPUs to the container
                # while limiting the visible GPUs to vLLM
                "--gpus", "all",
                "-e", "CUDA_VISIBLE_DEVICES=" + gpu_str,
                "--ipc", "host",
                "--pid", "host",
                "--net", "host",
                "--name", container_name,
                "-e", f"HF_TOKEN={hf_token}",
                "-e", f"LOG_LEVEL={log_level}",
                "-v", f"{hf_home}:/root/.cache/huggingface",
                *(
                    ["-v", f"{vllm_cache_dir}:/root/.cache/vllm/torch_compile_cache"] if vllm_cache_dir else []
                ),
                server_image,
                "--port", str(decode_http_base_port + i),
                "--model", model_id,
                "--tensor-parallel-size", str(len(cur_gpus)),
                "--gpu-memory-utilization", "0.8",
                "--trust-remote-code",
                "--max-num-seqs", str(max_num_seqs),
                "--kv-transfer-config", json.dumps(decode_kv_transfer_config),
            ]
            # fmt: on
            if max_num_batched_tokens is not None:
                server_cmd.extend(
                    ["--max-num-batched-tokens", str(max_num_batched_tokens)]
                )

            logger.info("Spawning vLLM server with command: %s", " ".join(server_cmd))
            logger.info("vLLM container name: %s", container_name)
            decode_log_filepath = server_log_filepath.with_name(
                f"decode_{i}_server_log.txt"
            )
            logger.info("vLLM logs will be written to %s", decode_log_filepath)
            decode_server_log_file = open(decode_log_filepath, "w")
            subprocess.Popen(
                server_cmd, stdout=decode_server_log_file, stderr=decode_server_log_file
            )
            spawned_containers.append(container_name)

    elif num_prefills == 0 and num_decodes == 0:
        # Single vLLM server
        gpu_str = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        gpu_str = f'"device={gpu_str}"'
        container_name = f"benchmark-vllm-{''.join(str(gpu_id) for gpu_id in gpu_ids)}"

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
            *(
                ["-v", f"{vllm_cache_dir}:/root/.cache/vllm/torch_compile_cache"] if vllm_cache_dir else []
            ),
            server_image,
            "--port", str(port),
            "--model", model_id,
            "--tensor-parallel-size", str(len(gpu_ids)),
            "--gpu-memory-utilization", "0.95",
            "--trust-remote-code",
            "--max-num-seqs", str(max_num_seqs),
        ]
        # fmt: on

        if max_num_batched_tokens is not None:
            server_cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

        logger.info("Spawning vLLM server with command: %s", " ".join(server_cmd))
        logger.info("vLLM container name: %s", container_name)
        logger.info("vLLM logs will be written to %s", server_log_filepath)
        server_log_file = open(server_log_filepath, "w")
        subprocess.Popen(server_cmd, stdout=server_log_file, stderr=server_log_file)
        spawned_containers.append(container_name)
    else:
        raise ValueError("Both num_prefills and num_decodes must be 0 or non-zero.")

    def kill_server():
        """Kill the vLLM servers."""
        for container_name in spawned_containers:
            logger.info("Killing vLLM server container %s", container_name)
            subprocess.run(["docker", "rm", "-f", container_name])
            logger.info("vLLM server container %s killed and removed", container_name)
        for handle in spawned_processes:
            handle.terminate()
            handle.kill()
            handle.wait()

    atexit.register(kill_server)

    return spawned_containers


def set_ulimit(target_soft_limit=10000):
    """Set the soft limit for the number of open files (ulimit -n)."""

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


def main(args: Args) -> None:
    logger.info("%s", args)

    assert isinstance(args.workload, WorkloadConfig)

    set_ulimit()

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
    # Optional envs
    vllm_cache_dir = os.environ.get("VLLM_CACHE_DIR", None)

    model_id = args.workload.model_id
    random.seed(args.workload.seed)
    np.random.seed(args.workload.seed)

    port = 8000 + int(cuda_visible_devices.split(",")[0])
    discovery_port = 30000 + int(cuda_visible_devices.split(",")[0])
    endpoint = {
        "openai": "/v1/completions",
        "openai-chat": "/v1/chat/completions",
    }[args.endpoint_type]
    api_url = f"http://127.0.0.1:{port}{endpoint}"

    # Kick off server startup
    logger.info("Spawning vLLM server...")
    container_names = spawn_vllm(
        server_image=args.server_image,
        port=port,
        discovery_port=discovery_port,
        model_id=model_id,
        hf_token=hf_token,
        hf_home=hf_home,
        gpu_ids=[int(gpu_id) for gpu_id in cuda_visible_devices.split(",")],
        max_num_seqs=args.workload.max_num_seqs,
        max_num_batched_tokens=args.workload.max_num_batched_tokens,
        log_level="INFO",
        server_log_filepath=args.workload.to_path(of="server_log"),
        vllm_cache_dir=vllm_cache_dir,
        num_prefills=args.workload.num_prefills if args.workload.num_prefills else 0,
        num_decodes=args.workload.num_decodes if args.workload.num_decodes else 0,
    )
    logger.info("Started vLLM containers %s", container_names)

    # Zeus
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        show_env()
    logger.info("Zeus environment information:\n%s", buffer.getvalue())
    zeus_monitor = ZeusMonitor()

    # Load the dataset. On its first time, it'll overlap nicely with vLLM startup.
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

    # Wait until the /health endpoint returns 200 OK
    health_url = f"http://127.0.0.1:{port}/health"
    logger.info("Waiting for vLLM server to become healthy at %s", health_url)
    tail_handle: subprocess.Popen | None = None
    if args.workload.num_prefills == 0 or args.workload.num_decodes == 0:
        tail_handle = subprocess.Popen(
            ["tail", "-f", args.workload.to_path(of="server_log")]
        )
    try:
        elapsed_seconds = 0
        while True:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is healthy.")
                    break
            except requests.RequestException as e:
                logger.warning("Waiting for vLLM server to become healthy: %s", e)

            time.sleep(1)
            elapsed_seconds += 1

            # Check the container state
            if elapsed_seconds >= 30:
                for container_name in container_names:
                    try:
                        container_running = (
                            subprocess.check_output(
                                [
                                    "docker",
                                    "inspect",
                                    "--format='{{json .State.Running}}'",
                                    container_name,
                                ]
                            )
                            .decode()
                            .strip()
                        )
                    except subprocess.CalledProcessError as e:
                        logger.exception(
                            "Failed to inspect container %s: %s", container_name, e
                        )
                        raise

                    if container_running != "'true'":
                        logger.error(
                            "Container %s seems to have crashed. Check server logs for details: %s",
                            container_name,
                            args.workload.to_path(of="server_log"),
                        )
                        raise RuntimeError(
                            f"Container {container_name} seems to have crashed."
                        )
    finally:
        if tail_handle is not None:
            tail_handle.terminate()
            tail_handle.wait()

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()

    benchmark_result = asyncio.run(
        benchmark(
            zeus_monitor=zeus_monitor,
            max_num_seqs=args.workload.max_num_seqs,
            workload=args.workload,
            endpoint_type=args.endpoint_type,
            api_url=api_url,
            model_id=model_id,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            max_output_tokens=args.max_output_tokens,
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
    result_json["num_prefills"] = args.workload.num_prefills
    result_json["num_decodes"] = args.workload.num_decodes

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

    # Something failed. Treat the whole run as a failure.
    if benchmark_result["completed"] < args.workload.num_requests:
        raise RuntimeError(
            f"Only {benchmark_result['completed']} out of {args.workload.num_requests} requests completed successfully. "
            "Raising RuntimeError to indicate failure."
        )


if __name__ == "__main__":
    args = tyro.cli(
        Args[
            ImageChat
            | VideoChat
            | AudioChat
            | OmniChat
            | LMArenaChat
            | GPQA
            | LengthControl
        ]
    )

    if (
        args.endpoint_type == "openai-chat"
        and args.workload.num_prefills
        and args.workload.num_decodes
    ):
        raise NotImplementedError(
            "vLLM OpenAI chat endpoint has a bug with Prefill/Decode disaggregation."
        )
    assert isinstance(args.workload, WorkloadConfig)

    # Exit if the result file exists so that the script is idempotent.
    result_file = args.workload.to_path(of="results")
    if result_file.exists() and not args.overwrite_results:
        logger.info(
            "Result file %s already exists. Exiting immediately. "
            "Specify --overwrite-results to run the benchmark and overwrite results.",
            result_file,
        )
        raise SystemExit(0)

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

    # Explicitly include Zeus PowerMonitor logs
    zl = logging.getLogger("zeus.monitor.power")
    zl.setLevel(logging.INFO)
    zl.propagate = True
    zl.handlers.clear()

    try:
        main(args)
    except Exception as e:
        logger.exception("An error occurred during the benchmark: %s", e)
        raise
