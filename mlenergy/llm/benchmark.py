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
from typing import Any, Generic, Literal, TypeVar, TYPE_CHECKING
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import tyro
import numpy as np
import requests
from pydantic import BaseModel
from zeus.monitor import ZeusMonitor, PowerMonitor, TemperatureMonitor
from zeus.show_env import show_env

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from mlenergy.llm.datasets import SampleRequest
from mlenergy.llm.workloads import (
    AudioChat,
    GPQA,
    ImageChat,
    LMArenaChat,
    SourcegraphFIM,
    OmniChat,
    VideoChat,
    WorkloadConfig,
    LengthControl,
)
from mlenergy.llm.config import (
    get_vllm_config_path,
    load_env_vars,
    load_extra_body,
    load_system_prompt,
)
from mlenergy.utils.container_runtime import (
    CleanupHandle,
    ContainerRuntime,
    DockerRuntime,
    SingularityRuntime,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

logger = logging.getLogger("mlenergy.llm.benchmark")


WorkloadT = TypeVar("WorkloadT", bound=WorkloadConfig)


class Args(BaseModel, Generic[WorkloadT]):
    """Data model for benchmark arguments.

    Attributes:
        workload: Workload configuration for the benchmark.
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
        server_image: Container image for the vLLM server. For Docker, use image names
            like "vllm/vllm-openai:v0.11.1". For Singularity, use .sif file paths.
        container_runtime: Container runtime to use ("docker" or "singularity").
        just_server: If set, only launch the server without running the benchmark.
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
    server_image: str = "vllm/vllm-openai:v0.11.1"
    container_runtime: Literal["docker", "singularity"] = "docker"
    just_server: bool = False

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
    """The input for the request function.

    Some models/tasks require pre-tokenized inputs. In that case, `prompt_token_ids`
    holds the list of token IDs for the prompt, and the completions API will use that.
    """

    prompt: str | list[str]
    api_url: str
    prompt_len: int
    output_len: int | None
    model: str
    prompt_token_ids: list[int] | None = None
    extra_body: dict | None = None
    multimodal_contents: list[dict] | None = None
    ignore_eos: bool = False
    system_prompt: str | None = None


@dataclass
class RequestFuncOutput:
    """The output of the request function including metrics."""

    prompt: str | list[str] = ""
    output_text: str = ""
    reasoning_output_text: str = ""
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
    client: aiohttp.ClientSession,
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

    if request_func_input.system_prompt:
        raise ValueError(
            "OpenAI Completions API does not support system prompt. "
            "Use OpenAI Chat Completions API instead."
        )

    payload = {
        "model": request_func_input.model,
        "prompt": request_func_input.prompt,
        "stop": [
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
            "<|fim_pad|>",
            "<|repo_name|>",
            "<|file_sep|>",
            "<|file_separator|>",
            "<|endoftext|>",
            "<|im_end|>",
        ],
        "repetition_penalty": 1.1,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
    }

    # Special case for models that require pre-tokenization.
    # Codestral is an example, because FIM prompts are not tokenized properly by vLLM.
    if request_func_input.prompt_token_ids is not None:
        payload["prompt"] = request_func_input.prompt_token_ids
        # payload["return_token_ids"] = True

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

    output_text = ""
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        async with client.post(url=api_url, json=payload, headers=headers) as response:
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
                                request_tracker.notify_tokens_generated(1)
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                usage = data.get("usage")
                                completion_tokens = usage and usage.get(
                                    "completion_tokens"
                                )
                                output.itl.append(timestamp - most_recent_timestamp)

                                if completion_tokens:
                                    inc = completion_tokens - current_completion_tokens
                                    # if inc == 0, below are no-ops
                                    request_tracker.notify_tokens_generated(inc)
                                    current_completion_tokens = completion_tokens
                                    for _ in range(inc - 1):
                                        output.itl.append(0)

                            most_recent_timestamp = timestamp
                            output_text += text or ""
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
                output.output_text = output_text
                output.latency = most_recent_timestamp - st
            else:
                output.error = (await response.text()) or response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
    finally:
        request_tracker.notify_request_finished()

    return output


async def async_request_openai_chat_completions(
    client: aiohttp.ClientSession,
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

    # Prepend system message if system prompt is provided
    if request_func_input.system_prompt:
        messages.insert(
            0, {"role": "system", "content": request_func_input.system_prompt}
        )

    payload = {
        "model": request_func_input.model,
        "messages": messages,
        "repetition_penalty": 1.1,
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

    output_text = ""
    reasoning_output_text = ""
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        async with client.post(url=api_url, json=payload, headers=headers) as response:
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
                        usage = data.get("usage")
                        completion_tokens = usage and usage.get("completion_tokens")
                        if completion_tokens == 0:
                            continue

                        if choices := data.get("choices"):
                            delta = choices[0]["delta"]
                            content = delta.get("content")
                            # Reasoning tokens are put in a separate field
                            # when a reasoning parser is specified.
                            reasoning_content = delta.get("reasoning_content")

                            # First token
                            if ttft == 0.0:
                                ttft = timestamp - st
                                request_tracker.notify_request_started()
                                request_tracker.notify_tokens_generated(1)
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                                if completion_tokens:
                                    inc = completion_tokens - current_completion_tokens
                                    # if inc == 0, below are no-ops
                                    request_tracker.notify_tokens_generated(inc)
                                    current_completion_tokens = completion_tokens
                                    for _ in range(inc - 1):
                                        output.itl.append(0)

                            if content is not None:
                                output_text += content
                            if reasoning_content is not None:
                                reasoning_output_text += reasoning_content
                        elif usage := data.get("usage"):
                            output.output_tokens = usage.get("completion_tokens")

                        most_recent_timestamp = timestamp

                output.output_text = output_text
                output.reasoning_output_text = reasoning_output_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = (await response.text()) or response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
    finally:
        request_tracker.notify_request_finished()
        if not output.success:
            logger.warning("Request failed with error: %s", output.error)

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
                        outputs[i].output_text + outputs[i].reasoning_output_text,
                        add_special_tokens=False,
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
    base_url: str,
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
    system_prompt: str | None,
):
    if endpoint_type in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[endpoint_type]
    else:
        raise ValueError(f"Unknown endpoint_type: {endpoint_type}")

    pool = aiohttp.TCPConnector(
        limit=0,
        ssl=False,
        keepalive_timeout=6 * 60 * 60,
    )
    client = aiohttp.ClientSession(
        timeout=AIOHTTP_TIMEOUT,
        connector=pool,
    )

    api_url = {
        "openai": f"{base_url}/v1/completions",
        "openai-chat": f"{base_url}/v1/chat/completions",
    }[endpoint_type]

    # Zeus power monitor
    power_monitor = PowerMonitor(update_period=0.1)
    temperature_monitor = TemperatureMonitor(update_period=0.5)

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
                client=client,
                request_func_input=request_func_input,
                request_tracker=request_tracker,
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
            prompt_token_ids=request.prompt_token_ids,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=output_len,
            multimodal_contents=request.multimodal_contents,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
            system_prompt=system_prompt,
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
    zeus_monitor.begin_window("steady_state", sync_execution=False)
    logger.info("Steady state has begun.")
    await request_tracker.wait_end()
    steady_state_end_time = time.time()
    steady_state_token_end = request_tracker.get_num_generated_tokens()
    steady_state_mes = zeus_monitor.end_window("steady_state", sync_execution=False)
    logger.info("Steady state finished.")
    steady_state_tokens = steady_state_token_end - steady_state_token_begin

    # Gather the rest of the requests.
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    logger.info("All requests have finished. Processing results...")
    # close the aiohttp client session
    await client.close()

    entire_mes = zeus_monitor.end_window("entire_benchmark", sync_execution=False)
    benchmark_end_time = time.time()
    benchmark_duration = benchmark_end_time - benchmark_start_time

    power_timeline = power_monitor.get_all_power_timelines(
        start_time=benchmark_start_time,
        end_time=benchmark_end_time,
    )
    temperature_timeline = temperature_monitor.get_temperature_timeline(
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
    steady_state_energy_per_token = steady_state_energy / steady_state_tokens
    # here we use the steady state energy per token to calculate per generation energy
    energy_per_generation = [
        steady_state_energy_per_token * output_len for output_len in actual_output_lens
    ]

    entire_benchmark_energy = sum(entire_mes.gpu_energy.values())
    entire_benchmark_energy_per_token = (
        entire_benchmark_energy / request_tracker.get_num_generated_tokens()
    )

    # fmt: off
    logger.info("{s:{c}^{n}}".format(s="Benchmark results", n=51, c="="))
    logger.info("%-40s: %d", "Total requests", workload.num_requests)
    logger.info("%-40s: %d", "Successful requests", metrics.completed)
    logger.info("%-40s: %.2f", "Benchmark total duration (s)", benchmark_duration)
    logger.info("%-40s: %.2f", "Benchmark total energy (J)", entire_benchmark_energy)
    logger.info("%-40s: %.2f", "Benchmark total energy (J) per token", entire_benchmark_energy_per_token)
    logger.info("%-40s: %d", "Steady state duration (s)", steady_state_time)
    if steady_state_energy is not None:
        logger.info("%-40s: %.2f", "Steady state energy (J)", steady_state_energy)
        logger.info("%-40s: %.2f", "Steady state average power (W)", steady_state_energy / steady_state_time)
    if steady_state_energy_per_token is not None:
        logger.info("%-40s: %.2f", "Steady state energy (J) per token", steady_state_energy_per_token)
        logger.info("%-40s: %.2f", "Average energy per generation (J)", steady_state_energy_per_token * metrics.total_output / metrics.completed)
    logger.info("%-40s: %d", "Total input tokens", metrics.total_input)
    logger.info("%-40s: %d", "Total generated tokens (usage stats)", metrics.total_output)
    logger.info("%-40s: %d", "Total generated tokens (counted)", request_tracker.get_num_generated_tokens())
    logger.info("%-40s: %.2f", "Request throughput (req/s)", metrics.request_throughput)
    logger.info("%-40s: %.2f", "Output token throughput (tok/s)", metrics.output_throughput)
    logger.info("%-40s: %.2f", "Total Token throughput (tok/s)", metrics.total_token_throughput)
    # fmt: on

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "steady_state_duration": steady_state_time,
        "steady_state_energy": steady_state_energy,
        "steady_state_energy_per_token": steady_state_energy_per_token,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "steady_state_measurement": asdict(steady_state_mes),
        "entire_benchmark_measurement": asdict(entire_mes),
        "timeline": {
            "benchmark_start_time": benchmark_start_time,
            "benchmark_end_time": benchmark_end_time,
            "steady_state_start_time": steady_state_start_time,
            "steady_state_end_time": steady_state_end_time,
            "power": power_timeline,
            "temperature": temperature_timeline,
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

    result["results"] = [
        {
            "prompt": output.prompt,
            "output_text": output.output_text,
            "reasoning_output_text": output.reasoning_output_text,
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
    ]

    return result


def spawn_vllm(
    runtime: ContainerRuntime,
    server_image: str,
    port: int,
    model_id: str,
    gpu_model: str,
    workload: str,
    hf_token: str,
    hf_home: str,
    gpu_ids: list[int],
    max_num_seqs: int,
    max_num_batched_tokens: int | None,
    server_log_filepath: Path,
    vllm_cache_dir: str | None = None,
) -> list[CleanupHandle]:
    """Spawn vLLM server(s).

    This does not wait for the server to be ready.

    Args:
        runtime: Container runtime instance

    Returns:
        Cleanup handles for the spawned servers (container names or process handles).
    """
    assert Path(hf_home).exists(), f"Hugging Face home directory not found: {hf_home}"

    spawned_cleanup_handles: list[CleanupHandle] = []

    # Monolithic deployment
    # Load model-specific config for monolithic deployment
    monolithic_config_path = get_vllm_config_path(
        model_id, gpu_model, workload, "monolithic"
    )
    monolithic_env_vars = load_env_vars(model_id, gpu_model, workload, "monolithic")

    # If the config file mentioned DP, we're doing that; otherwise TP.
    if "data-parallel-size" in monolithic_config_path.read_text():
        parallel_arg = "--data-parallel-size"
        max_num_seqs_per_replica, rem = divmod(max_num_seqs, len(gpu_ids))
        if rem != 0:
            raise ValueError(
                f"For data parallelism, max_num_seqs ({max_num_seqs}) must be "
                f"divisible by the number of GPUs ({len(gpu_ids)})."
            )
        max_num_seqs = max_num_seqs_per_replica
    else:
        parallel_arg = "--tensor-parallel-size"

    container_name = f"benchmark-vllm-{''.join(str(gpu_id) for gpu_id in gpu_ids)}"

    # Container path for the config file
    container_config_path = "/vllm_config/monolithic.config.yaml"

    # Build environment variables
    env_vars = {
        "HF_TOKEN": hf_token,
        "HF_HOME": hf_home,
        **monolithic_env_vars,
    }
    if vllm_cache_dir:
        env_vars["VLLM_CACHE_DIR"] = vllm_cache_dir

    # Build bind mounts
    # Use identity mapping (same path in container as host) to work with both
    # Docker (root user) and Singularity (runs as user with home mounted)
    bind_mounts = [
        (str(monolithic_config_path), container_config_path, "ro"),
        (hf_home, hf_home, ""),
    ]
    if vllm_cache_dir:
        bind_mounts.append((vllm_cache_dir, vllm_cache_dir, ""))

    # Build vLLM command
    vllm_cmd = [
        "vllm",
        "serve",
        model_id,
        "--config",
        container_config_path,
        "--port",
        str(port),
        parallel_arg,
        str(len(gpu_ids)),
        "--max-num-seqs",
        str(max_num_seqs),
    ]
    if max_num_batched_tokens is not None:
        vllm_cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

    # Build container run command
    server_cmd = runtime.build_run_command(
        image=server_image,
        container_name=container_name,
        gpu_ids=gpu_ids,
        env_vars=env_vars,
        bind_mounts=bind_mounts,
        command=vllm_cmd,
    )

    logger.info("Spawning vLLM server with command: %s", " ".join(server_cmd))
    logger.info("vLLM logs will be written to %s", server_log_filepath)
    server_log_file = open(server_log_filepath, "w")
    process_handle = subprocess.Popen(
        server_cmd, stdout=server_log_file, stderr=server_log_file
    )
    cleanup_handle = runtime.get_cleanup_handle(container_name, process_handle)
    spawned_cleanup_handles.append(cleanup_handle)

    def kill_server():
        """Kill the vLLM servers."""
        for cleanup_handle in spawned_cleanup_handles:
            cleanup_handle.cleanup()
        for cleanup_handle in spawned_cleanup_handles:
            cleanup_handle.wait()

    atexit.register(kill_server)

    return spawned_cleanup_handles


def set_ulimit(target_soft_limit=65535):
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
    base_url = f"http://127.0.0.1:{port}"
    logger.info("Server URL: %s", base_url)

    # Create container runtime instance
    if args.container_runtime == "docker":
        runtime = DockerRuntime()
        logger.info("Using Docker container runtime")
    elif args.container_runtime == "singularity":
        runtime = SingularityRuntime()
        logger.info("Using Singularity container runtime")
    else:
        raise ValueError(f"Unknown container runtime: {args.container_runtime}")

    # Kick off server startup
    logger.info("Spawning vLLM server...")
    cleanup_handles = spawn_vllm(
        runtime=runtime,
        server_image=args.server_image,
        port=port,
        model_id=model_id,
        gpu_model=args.workload.gpu_model,
        workload=args.workload.normalized_name,
        hf_token=hf_token,
        hf_home=hf_home,
        gpu_ids=[int(gpu_id) for gpu_id in cuda_visible_devices.split(",")],
        max_num_seqs=args.workload.max_num_seqs,
        max_num_batched_tokens=args.workload.max_num_batched_tokens,
        server_log_filepath=args.workload.to_path(of="server_log"),
        vllm_cache_dir=vllm_cache_dir,
    )
    logger.info("Started vLLM servers (cleanup handles: %s)", len(cleanup_handles))

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

    # Read in extra body if specified in the config directory
    extra_body = load_extra_body(
        model_id=args.workload.model_id,
        gpu_model=args.workload.gpu_model,
        workload=args.workload.normalized_name,
    )

    # Load system prompt if specified in the config directory
    system_prompt = load_system_prompt(
        model_id=args.workload.model_id,
        gpu_model=args.workload.gpu_model,
        workload=args.workload.normalized_name,
    )

    # Wait until the /health endpoint returns 200 OK
    health_url = f"http://127.0.0.1:{port}/health"
    logger.info("Waiting for vLLM server to become healthy at %s", health_url)

    # Monitor server log file size to detect if server has crashed
    server_log_path = args.workload.to_path(of="server_log")
    last_log_size = 0
    last_change_time = time.time()

    while True:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server is healthy.")
                break
        except requests.RequestException as e:
            logger.warning("Waiting for vLLM server to become healthy: %s", e)

        # Check if server log is still growing
        try:
            current_log_size = os.path.getsize(server_log_path)
            if current_log_size != last_log_size:
                last_log_size = current_log_size
                last_change_time = time.time()
            elif time.time() - last_change_time > 60:
                logger.error(
                    "Server log has not changed for 60 seconds. Server appears to have crashed. "
                    "Check server logs for details: %s",
                    server_log_path,
                )
                raise RuntimeError(
                    "Server appears to have crashed (log file unchanged for 60 seconds)."
                )
        except FileNotFoundError:
            pass

        time.sleep(1)

    if args.just_server:
        try:
            input("Press any key to terminate the server and exit...")
        finally:
            logger.info("Terminating the server...")
            return

    # Freeze gc to prevent random pauses during benchmarking
    gc.collect()
    gc.freeze()

    benchmark_result = asyncio.run(
        benchmark(
            zeus_monitor=zeus_monitor,
            max_num_seqs=args.workload.max_num_seqs,
            workload=args.workload,
            endpoint_type=args.workload.endpoint_type,
            base_url=base_url,
            model_id=model_id,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            max_output_tokens=args.max_output_tokens,
            max_concurrency=args.max_concurrency,
            extra_body=extra_body | sampling_params,
            system_prompt=system_prompt,
        )
    )

    # Save config and results to json
    result_json: dict[str, Any] = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["endpoint_type"] = args.workload.endpoint_type
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
            | SourcegraphFIM
            | GPQA
            | LengthControl
        ]
    )

    assert isinstance(args.workload, WorkloadConfig)

    # Exit if the result file exists so that the script is idempotent.
    result_file = args.workload.to_path(of="results")
    if result_file.exists() and not args.overwrite_results and not args.just_server:
        print(
            f"Result file {result_file} already exists. Exiting immediately. "
            "Specify --overwrite-results to run the benchmark and overwrite results.",
        )
        raise SystemExit(0)

    # Set up the logger so that it logs to both console and file
    if args.just_server:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s: %(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler()],
        )
    else:
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
        zpl = logging.getLogger("zeus.monitor.power")
        zpl.setLevel(logging.INFO)
        zpl.propagate = True
        zpl.handlers.clear()

        # Explicitly include Zeus TemperatureMonitor logs
        ztl = logging.getLogger("zeus.monitor.temperature")
        ztl.setLevel(logging.INFO)
        ztl.propagate = True
        ztl.handlers.clear()

    try:
        main(args)
    except Exception as e:
        logger.exception("An error occurred during the benchmark: %s", e)
        raise
