"""dLLM benchmark runner.

Models the LLM and diffusion benchmark runner, reuses the LMArenaChat workload
in `mlenergy.llm.workloads`
"""

from dataclasses import asdict
import json
import subprocess
import sys
import tyro
import logging

from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, TypeVar, Generic
from datetime import datetime


from mlenergy.llm.workloads import (
    LMArenaChat,
    WorkloadConfig,
)

from mlenergy.dllm.workloads import (
    LMArenaChatDLLM,
    default_lmarena_chat_dllm,
)

from mlenergy.dllm.dllm_runtime import (
    DLLMRuntime,
    FastDLLMRuntime,
    default_fast_dllm_runtime,
)


model_ids = ["GSAI-ML/LLaDA-8B-Instruct"]

logger = logging.getLogger("mlenergy.dllm.benchmark")

WorkloadT = TypeVar("WorkloadT", bound=WorkloadConfig)
DLLMRuntimeT = TypeVar("DLLMRuntimeT", bound=DLLMRuntime)


class DLLMArgs(BaseModel, Generic[WorkloadT, DLLMRuntimeT]):
    """Arguments for dLLM benchmark runner.

    Attributes:
        workload: Workload configuration for dLLM: {LMArenaChatDLLM}
        dllm_runtime: Runtime configuration for dLLM: {FastDLLMRuntime}
        warmup_iters: Number of warmup iterations.
        benchmark_iters: Number of benchmark iterations.
    """

    workload: WorkloadT = Field(default_factory=default_lmarena_chat_dllm)
    dllm_runtime: DLLMRuntimeT = Field(default_factory=default_fast_dllm_runtime)
    warmup_iters: int = 2
    benchmark_iters: int = 6


def save_results(
    args: DLLMArgs,
    benchmark_duration: float,
    total_energy_result: Any = None,
    iter_energy_results: list[Any] = None,
) -> None:
    # Calculate metrics
    num_images = args.workload.batch_size * args.benchmark_iters

    # Prepare results dictionary
    result_json: dict[str, Any] = {}

    # Setup information
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["model_id"] = args.workload.model_id
    result_json["batch_size"] = args.workload.batch_size
    result_json["num_steps"] = args.benchmark_iters

    # TODO: Performance metrics

    # Energy metrics
    if total_energy_result:
        result_json["total_energy"] = total_energy_result.total_energy
        result_json["energy_measurement"] = asdict(total_energy_result)

    for i, iter_energy_result in enumerate(iter_energy_results):
        result_json[f"iter{i}_energy_measurement"] = asdict(iter_energy_result)

    # TODO: Configuration details

    # Save results
    result_file = args.workload.to_path(of="results")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    logger.info(f"Results saved to {result_file}")

    # Log summary
    logger.info("[DLLM Benchmark Results]")
    logger.info("%-40s: %s", "Model ID", args.workload.model_id)
    logger.info("%-40s: %d", "Batch Size", args.workload.batch_size)
    logger.info("%-40s: %d", "Total Images Requested", num_images)
    logger.info("%-40s: %.2f", "Total Time (s)", benchmark_duration)

    if total_energy_result:
        logger.info("%-40s: %.2f", "Total Energy (J)", total_energy_result.total_energy)
        logger.info(
            "%-40s: %.2f",
            "Energy per Image (J)",
            total_energy_result.total_energy / num_images,
        )


def main(args: DLLMArgs) -> None:
    """Main benchmark function.

    Args:
        args: Benchmark arguments with workload and runtime configurations.
    """
    logger.info("Installing dLLM runtime: %s", type(args.dllm_runtime).__name__)
    args.dllm_runtime.install_runtime()


if __name__ == "__main__":
    args = tyro.cli(
        DLLMArgs[
            LMArenaChatDLLM,
            FastDLLMRuntime,
        ]
    )

    # Set up logging
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
