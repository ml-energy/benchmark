"""Diffusion model benchmark runner.

Similar to LLM benchmark, this provides comprehensive benchmarking
for various diffusion models with energy monitoring and detailed metrics.
"""

from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import os
import random
import sys
import time
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import torch
import torch.distributed
import tyro
from pydantic import BaseModel
from transformers.models.t5 import T5EncoderModel
from zeus.monitor import ZeusMonitor
from zeus.show_env import show_env

from xfuser import (
    xFuserArgs,
    xFuserFluxPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserHunyuanDiTPipeline,
    xFuserArgs
)
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
)

from mlenergy.diffusion.dataset import DiffusionRequest
from mlenergy.diffusion.workloads import (
    DiffusionWorkloadConfig,
    TextToImage,
    TextToVideo,
)

logger = logging.getLogger("mlenergy.diffusion.run")

WorkloadT = TypeVar("WorkloadT", bound=DiffusionWorkloadConfig)

# Pipeline configurations - map model_id to pipeline
PIPELINE_CONFIGS = {
    "black-forest-labs/FLUX.1-dev": {
        "pipeline_class": xFuserFluxPipeline,
        "needs_t5": True,
        "t5_subfolder": "text_encoder_2",
        "dtype": torch.bfloat16,
    },
    "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS": {
        "pipeline_class": xFuserPixArtSigmaPipeline,
        "needs_t5": True,
        "t5_subfolder": "text_encoder",
        "dtype": torch.bfloat16,
    },
    "stabilityai/stable-diffusion-3-medium-diffusers": {
        "pipeline_class": xFuserStableDiffusion3Pipeline,
        "needs_t5": True,
        "t5_subfolder": "text_encoder_3",
        "dtype": torch.bfloat16,
    },
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers": {
        "pipeline_class": xFuserHunyuanDiTPipeline,
        "needs_t5": True,
        "t5_subfolder": "text_encoder_2",
        "dtype": torch.bfloat16,
    },
}


class DiffusionArgs(BaseModel, Generic[WorkloadT]):
    """Data model for diffusion benchmark arguments.

    Attributes:
        workload: Workload configuration for the benchmark.
        overwrite_results: Whether to overwrite existing results.
        save_images: Whether to save generated images.
        ulysses_degree: Ulysses attention parallelism degree.
        ring_degree: Ring attention parallelism degree.
        use_torch_compile: Whether to use torch.compile.
        use_fp8_t5_encoder: Whether to use FP8 quantization for T5.
        enable_sequential_cpu_offload: Whether to use CPU offload.
    """

    # Workload configuration
    workload: WorkloadT
    warmup_iters: int = 2
    benchmark_iters: int = 4

    # Results configuration
    overwrite_results: bool = False
    save_images: bool = True

    # Parallelism configuration
    ulysses_degree: int = 1
    ring_degree: int = 1


def get_model_type_from_id(model_id: str) -> str:
    """Get model type from model_id for backward compatibility."""
    if "FLUX" in model_id:
        return "Flux"
    elif "stable-diffusion-xl" in model_id:
        return "SDXL"
    elif "PixArt-alpha" in model_id and "Sigma" not in model_id:
        return "Pixart-alpha"
    elif "PixArt-Sigma" in model_id:
        return "Pixart-sigma"
    elif "stable-diffusion-3" in model_id:
        return "Sd3"
    elif "HunyuanDiT" in model_id:
        return "HunyuanDiT"
    else:
        return "Unknown"


def setup_pipeline(
    model_id: str,
    engine_config: Any,
    local_rank: int,
    args: DiffusionArgs
) -> tuple[Any, int]:
    """Setup the appropriate pipeline based on model_id."""
    if model_id not in PIPELINE_CONFIGS:
        raise ValueError(f"Unsupported model_id: {model_id}")
    
    config = PIPELINE_CONFIGS[model_id]

    cache_args = {
        "use_teacache": False,
        "use_fbcache": False,
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": args.workload.inference_steps,
    }
    
    # Handle T5 encoder if needed
    text_encoder_kwargs = {}
    if config["needs_t5"]:
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder=config["t5_subfolder"],
            torch_dtype=config["dtype"]
        )

        text_encoder_kwargs[config["t5_subfolder"]] = text_encoder
    
    # if args.use_fp8_t5_encoder:
    #     try:
    #         from optimum.quanto import freeze, qfloat8, quantize
    #         logging.info(f"rank {local_rank} quantizing text encoder 2")
    #         quantize(text_encoder, weights=qfloat8)
    #         freeze(text_encoder)
    #     except ImportError:
    #         logging.warning("optimum.quanto not available, skipping T5 quantization")

    # Initialize pipeline
    pipeline_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "engine_config": engine_config,
        "torch_dtype": config["dtype"],
        **text_encoder_kwargs
    }
    
    # Add cache args for models that support it
    if get_model_type_from_id(model_id) == "Flux":
        pipeline_kwargs["cache_args"] = cache_args
    
    pipe = config["pipeline_class"].from_pretrained(**pipeline_kwargs)

    # Handle device placement
    # if args.enable_sequential_cpu_offload:
    #     pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
    #     logger.info(f"rank {local_rank} sequential CPU offload enabled")
    # else:
    pipe = pipe.to(f"cuda:{local_rank}")

    return pipe


def get_inference_kwargs(
    request: DiffusionRequest,
    guidance_scale: float,
    args: DiffusionArgs,
) -> Any:
    inference_kwargs = {
        "prompt": request.prompts,
        "height": args.workload.height,
        "width": args.workload.width,
        "num_inference_steps": args.workload.inference_steps,
        "guidance_scale": guidance_scale,
        "output_type": "pil" if args.save_images else "latent",
        "generator": torch.Generator(device="cuda").manual_seed(args.workload.seed),
    }
    
    model_type = get_model_type_from_id(args.workload.model_id)
    if model_type == "Flux":
        inference_kwargs["max_sequence_length"] = 256
    elif model_type in ["Pixart-alpha", "Pixart-sigma", "HunyuanDiT"]:
        inference_kwargs["use_resolution_binning"] = True
    
    # Add video parameters
    if hasattr(args.workload, 'num_frames'):
        inference_kwargs["num_frames"] = args.workload.num_frames
    if hasattr(args.workload, 'fps'):
        inference_kwargs["fps"] = args.workload.fps
    
    return inference_kwargs


def save_generated_images(
    pipe: Any,
    output: Any,
    request: Any,
    args: DiffusionArgs,
    output_dir: Path,
    iteration_idx: int = 0
):
    if args.save_images:
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        num_prompts = len(request.prompts)
        dp_batch_size = (num_prompts + num_dp_groups - 1) // num_dp_groups
        
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                if image_rank < num_prompts:
                    prompt_text = request.prompts[image_rank]
                    # Extract a clean name from the prompt
                    words = prompt_text.split()[:5]  # Take first 5 words
                    safe_name = "_".join(word for word in words if word.isalnum() or word in ['-', '_'])[:30]
                    safe_name = safe_name.replace(' ', '_').lower()
                    filename = f"i{iteration_idx}-{image_rank}_{safe_name}.png"
                    image_path = output_dir / filename
                    image.save(image_path)
                    logger.info(f"Saved image {image_rank} of {iteration_idx} to {image_path}")


def save_results(
    args: DiffusionArgs,
    benchmark_duration: float,
    total_energy_result: Any = None,
    iter_energy_results: list[Any] = None,
    local_rank: int = 0
) -> None:
    if local_rank != 0:
        return
        
    # Calculate metrics
    num_images = args.workload.batch_size * args.benchmark_iters
    throughput = num_images / benchmark_duration if benchmark_duration > 0 else 0.0
    avg_time_per_image = benchmark_duration / num_images if num_images > 0 else 0.0
    
    # Prepare results dictionary
    result_json: dict[str, Any] = {}
    
    # Setup information
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["model_id"] = args.workload.model_id
    result_json["batch_size"] = args.workload.batch_size
    result_json["num_iterations"] = args.benchmark_iters
    
    # Generation parameters
    result_json["height"] = args.workload.height
    result_json["width"] = args.workload.width 
    result_json["inference_steps"] = args.workload.inference_steps
    result_json["seed"] = args.workload.seed
    
    # Performance metrics  
    result_json["total_images"] = num_images
    result_json["total_time"] = benchmark_duration
    result_json["throughput_images_per_sec"] = throughput
    result_json["avg_time_per_image"] = avg_time_per_image
    
    # Energy metrics
    if total_energy_result:
        result_json["total_energy"] = total_energy_result.total_energy
        result_json["energy_per_image"] = total_energy_result.total_energy / num_images if num_images > 0 else 0.0
        result_json["energy_measurement"] = asdict(total_energy_result)
    
    for i, iter_energy_result in enumerate(iter_energy_results):
        result_json[f"iter{i}_energy_measurement"] = asdict(iter_energy_result)
    
    # Configuration details
    result_json["configurations"] = {
        "ulysses_degree": args.ulysses_degree,
        "ring_degree": args.ring_degree,
    }

    # Save results
    result_file = args.workload.to_path(of="results")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)
    
    logger.info(f"Results saved to {result_file}")
    
    # Log summary
    logger.info("[Diffusion Benchmark Results]")
    logger.info("%-40s: %s", "Model ID", args.workload.model_id)
    logger.info("%-40s: %d", "Batch Size", args.workload.batch_size)
    logger.info("%-40s: %d", "Total Images Requested", num_images)
    logger.info("%-40s: %.2f", "Total Time (s)", benchmark_duration)
    logger.info("%-40s: %.2f", "Images per Second", throughput)
    logger.info("%-40s: %.2f", "Seconds per Image", avg_time_per_image)
    
    if total_energy_result:
        logger.info("%-40s: %.2f", "Total Energy (J)", total_energy_result.total_energy)
        logger.info("%-40s: %.2f", "Energy per Image (J)", total_energy_result.total_energy / num_images)


def main(args: DiffusionArgs) -> None:
    """Main benchmark function."""
    logger.info("%s", args)
    assert isinstance(args.workload, DiffusionWorkloadConfig)

    result_file = args.workload.to_path(of="results")
    if result_file.exists() and not args.overwrite_results:
        logger.info(
            "Result file %s already exists. Exiting immediately. "
            "Specify --overwrite_results to run the benchmark and overwrite results.",
            result_file,
        )
        return
    
    # # Necessary envs
    # cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    # hf_token = os.environ["HF_TOKEN"]
    # hf_home = os.environ["HF_HOME"]
    
    zeus_monitor = None
    if os.environ.get("LOCAL_RANK", "0") == "0":
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            show_env()
        logger.info("Zeus environment information:\n%s", buffer.getvalue())

        zeus_monitor = ZeusMonitor()
    
    random.seed(args.workload.seed)
    np.random.seed(args.workload.seed)
    torch.manual_seed(args.workload.seed)

    requests = args.workload.load_requests(args.warmup_iters, args.benchmark_iters)

    # Setup xFuser arguments
    logger.info(f"Setting up xFuser args")
    xfuser_args = xFuserArgs(
        model=args.workload.model_id,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        height=args.workload.height,
        width=args.workload.width,
        num_inference_steps=args.workload.inference_steps,
        seed=args.workload.seed,
        prompt=requests[0].prompts,
        use_torch_compile=args.workload.use_torch_compile,
    )
    engine_config, input_config = xfuser_args.create_config()

    world_group = get_world_group()
    local_rank = world_group.local_rank
    world_size = world_group.world_size

    # Setup xFuser pipeline
    logger.info(f"Setting up xFuser pipeline")
    pipe = setup_pipeline(args.workload.model_id, engine_config, local_rank, args)

    logger.info(f"Preparing xFuser pipeline")
    pipe.prepare_run(input_config, steps=args.workload.inference_steps)

    # Warmup iterations with different requests
    logger.info(f"Running {args.warmup_iters} warmup iterations with different requests")
    output_dir = args.workload.to_path(of="image_outputs")
    
    for i in range(args.warmup_iters):
        logger.info(f"Warmup iteration {i+1}/{args.warmup_iters}")
        warmup_request = requests[i]
        warmup_kwargs = get_inference_kwargs(warmup_request, input_config.guidance_scale, args)
        warmup_output = pipe(**warmup_kwargs)
        # save_generated_images(pipe, warmup_output, warmup_request, args, output_dir, i)

    # Benchmark iterations with different requests
    logger.info(f"Start running {args.benchmark_iters} benchmark iterations")
    iter_energy_results = []
    torch.cuda.synchronize()
    torch.distributed.barrier()
    benchmark_start_time = time.perf_counter()
    
    if zeus_monitor:
        zeus_monitor.begin_window("entire_benchmark")
    
    for i in range(args.benchmark_iters):
        request_idx = args.warmup_iters + i
        benchmark_request = requests[request_idx]
        benchmark_kwargs = get_inference_kwargs(benchmark_request, input_config.guidance_scale, args)
        
        logger.info(f"Benchmark iteration {i+1}/{args.benchmark_iters}")
        if zeus_monitor:
            zeus_monitor.begin_window("iteration")
        benchmark_output = pipe(**benchmark_kwargs)
        if zeus_monitor:
                iter_energy_results.append(zeus_monitor.end_window("iteration"))

        save_generated_images(pipe, benchmark_output, benchmark_request, args, output_dir, i)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    total_energy_result = None
    if zeus_monitor:
        total_energy_result = zeus_monitor.end_window("entire_benchmark")
    
    benchmark_duration = time.perf_counter() - benchmark_start_time
    logger.info(f"End running diffusion benchmark, duration: {benchmark_duration:.2f}s")

    if local_rank == 0:
        save_results(args, benchmark_duration, total_energy_result, iter_energy_results)
    
    get_runtime_state().destroy_distributed_env()


# TODO: download the model if not exists
# TODO: handle the default configs of different models
# TODO: move SP degree to the workload config
# TODO: handle server log
if __name__ == "__main__":
    args = tyro.cli(DiffusionArgs[TextToImage | TextToVideo])

    # Set up logging
    # Only rank 0 should write to the driver log file to avoid conflicts
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # Create handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s: %(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)
    
    handlers = [stream_handler]
    
    if local_rank == 0:
        file_handler = logging.FileHandler(args.workload.to_path(of="driver_log"), mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger with force=True to override any existing configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s: %(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    try:
        main(args)
    except Exception as e:
        logger.exception("An error occurred during the benchmark: %s", e)
        raise 