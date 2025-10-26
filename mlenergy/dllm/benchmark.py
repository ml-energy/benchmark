"""dLLM benchmark runner.

Models the LLM and diffusion benchmark runner, reuses the LMArenaChat workload
in `mlenergy.llm.workloads`
"""

import json
import time
import logging
from dataclasses import asdict
from typing import Any, TypeVar, Generic
from datetime import datetime

import tyro
from pydantic import BaseModel, Field


from mlenergy.llm.workloads import (
    WorkloadConfig,
)
from mlenergy.dllm.workloads import (
    LMArenaChatDLLM,
    default_lmarena_chat_dllm,
)
from mlenergy.dllm.dllm_runtime import (
    DLLMRuntime,
    LladaRuntime,
    DreamRuntime,
    default_llada_runtime,
)
from zeus.monitor import ZeusMonitor, PowerMonitor, TemperatureMonitor


model_ids = ["GSAI-ML/LLaDA-8B-Instruct"]

logger = logging.getLogger(__name__)

WorkloadT = TypeVar("WorkloadT", bound=WorkloadConfig)
DLLMRuntimeT = TypeVar("DLLMRuntimeT", bound=DLLMRuntime)


class DLLMArgs(BaseModel, Generic[WorkloadT, DLLMRuntimeT]):
    """Arguments for dLLM benchmark runner.

    Attributes:
        workload: Workload configuration for dLLM: {LMArenaChatDLLM}
        dllm_runtime: Runtime configuration for dLLM: {LladaRuntime, DreamRuntime}
        warmup_iters: Number of warmup iterations.
        benchmark_iters: Number of benchmark iterations.
    """

    workload: WorkloadT = Field(default_factory=default_lmarena_chat_dllm)
    dllm_runtime: DLLMRuntimeT = Field(default_factory=default_llada_runtime)
    warmup_iters: int = 2
    benchmark_iters: int = 100

    def model_post_init(self, __context):
        super().model_post_init(__context)
        num_workload_iters = self.workload.num_requests // self.workload.batch_size
        if self.warmup_iters > num_workload_iters:
            raise ValueError(
                f"warmup_iters ({self.warmup_iters}) cannot be greater than "
                f"the number of workload iterations ({num_workload_iters})"
            )
        if self.benchmark_iters > num_workload_iters:
            logger.warning(
                f"benchmark_iters ({self.benchmark_iters}) is greater than "
                f"the number of workload iterations ({num_workload_iters})"
                f". Setting benchmark_iters to {num_workload_iters}."
            )
            self.benchmark_iters = num_workload_iters


def save_results(
    args: DLLMArgs,
    benchmark_duration: float,
    benchmark_start_time: float,
    benchmark_end_time: float,
    total_energy_result: Any,
    iter_energy_results: list[Any],
    power_timeline: Any,
    temperature_timeline: Any,
) -> None:
    result_json: dict[str, Any] = {}

    # General config
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["model_id"] = args.workload.model_id
    result_json["batch_size"] = args.workload.batch_size
    result_json["num_iters"] = args.benchmark_iters

    # dLLM Runtime configuration
    result_json["runtime_type"] = type(args.dllm_runtime).__name__
    result_json["runtime_config"] = {
        "model_id": args.dllm_runtime.model_id,
        "steps": args.dllm_runtime.steps,
        "gen_length": args.dllm_runtime.gen_length,
        "block_length": args.dllm_runtime.block_length,
        "cache_mode": args.dllm_runtime.cache_mode,
        "remasking": args.dllm_runtime.remasking,
    }

    # Energy, power, temperature
    result_json["total_energy"] = total_energy_result.total_energy
    result_json["energy_measurement"] = asdict(total_energy_result)

    for i, iter_energy_result in enumerate(iter_energy_results):
        result_json[f"iter{i}_energy_measurement"] = asdict(iter_energy_result)

    result_json["timeline"] = {
        "benchmark_start_time": benchmark_start_time,
        "benchmark_end_time": benchmark_end_time,
        "power": power_timeline,
        "temperature": temperature_timeline,
    }

    result_file = args.workload.to_path(of="results")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    logger.info(f"Results saved to {result_file}")

    # Log summary
    logger.info("[DLLM Benchmark Results]")
    logger.info("%-40s: %s", "Model ID", args.workload.model_id)
    logger.info("%-40s: %d", "Batch Size", args.workload.batch_size)
    logger.info("%-40s: %.2f", "Total Time (s)", benchmark_duration)

    if total_energy_result:
        logger.info("%-40s: %.2f", "Total Energy (J)", total_energy_result.total_energy)


def main(args: DLLMArgs) -> None:
    """Main benchmark function.

    Args:
        args: Benchmark arguments with workload and runtime configurations.
    """
    zeus_monitor = ZeusMonitor()
    power_monitor = PowerMonitor(update_period=0.1)
    temperature_monitor = TemperatureMonitor(update_period=0.5)

    logger.info("Loading requests from workload: %s", type(args.workload).__name__)
    input_requests = args.workload.load_requests()
    logger.info("Loaded %d requests", len(input_requests))

    batch_size = args.workload.batch_size
    logger.info("Using batch size: %d", batch_size)

    args.dllm_runtime.load_model()

    for _ in range(args.warmup_iters):
        logger.info("Starting warmup iterations")
        batch_prompts = [
            req.prompt if isinstance(req.prompt, str) else req.prompt[-1]
            for req in input_requests[:batch_size]
        ]
        args.dllm_runtime.run_one_batch(batch_prompts)

    logger.info("Warmup complete. Starting benchmark iterations...")
    zeus_monitor.begin_window("total")
    iter_energy_results = []
    # We will restart from the beginning when doing benchmark runs
    # So batch_idx starts from 0
    num_batches = (len(input_requests) + batch_size - 1) // batch_size
    benchmark_start_time = time.time()
    for batch_idx in range(num_batches):
        zeus_monitor.begin_window("iter")
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(input_requests))

        batch_requests = input_requests[start_idx:end_idx]
        batch_prompts = [
            req.prompt if isinstance(req.prompt, str) else req.prompt[-1]
            for req in batch_requests
        ]

        logger.info(
            "Processing batch %d/%d (requests %d-%d)",
            batch_idx + 1,
            num_batches,
            start_idx,
            end_idx - 1,
        )

        outputs = args.dllm_runtime.run_one_batch(batch_prompts)

        for i, (req, output) in enumerate(zip(batch_requests, outputs)):
            logger.debug(
                "Request %d - Prompt length: %d, Expected output length: %d",
                start_idx + i,
                req.prompt_len,
                req.expected_output_len,
            )
            logger.debug("Generated output preview: %s...", output[:100])
        iter_energy_results.append(zeus_monitor.end_window("iter"))

    benchmark_end_time = time.time()
    zeus_metrics = zeus_monitor.end_window("total")
    benchmark_duration = benchmark_end_time - benchmark_start_time
    power_timeline = power_monitor.get_all_power_timelines(
        start_time=benchmark_start_time,
        end_time=benchmark_end_time,
    )
    temperature_timeline = temperature_monitor.get_temperature_timeline(
        start_time=benchmark_start_time,
        end_time=benchmark_end_time,
    )
    save_results(
        args,
        benchmark_duration=benchmark_duration,
        benchmark_start_time=benchmark_start_time,
        benchmark_end_time=benchmark_end_time,
        total_energy_result=zeus_metrics,
        iter_energy_results=iter_energy_results,
        power_timeline=power_timeline,
        temperature_timeline=temperature_timeline,
    )

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    args = tyro.cli(
        DLLMArgs[
            LMArenaChatDLLM,
            LladaRuntime | DreamRuntime,
        ]
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s: %(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.workload.to_path(of="driver_log"), mode="w"),
        ],
    )

    main(args)
