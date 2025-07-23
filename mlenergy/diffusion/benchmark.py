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
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserHunyuanDiTPipeline,
    xFuserStableDiffusionXLPipeline,
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

# Model configurations - map model_id to pipeline and settings
MODEL_CONFIGS = {
    "black-forest-labs/FLUX.1-dev": {
        "pipeline_class": xFuserFluxPipeline,
        "inference_steps": 28,
        "needs_t5": True,
        "t5_subfolder": "text_encoder_2",
        "dtype": torch.bfloat16,
        "supports_video": False
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "pipeline_class": xFuserStableDiffusionXLPipeline,
        "inference_steps": 30,
        "needs_t5": False,
        "dtype": torch.float16,
        "supports_video": False
    },
    "PixArt-alpha/PixArt-XL-2-1024-MS": {
        "pipeline_class": xFuserPixArtAlphaPipeline,
        "inference_steps": 20,
        "needs_t5": False,
        "dtype": torch.float16,
        "supports_video": False
    },
    "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS": {
        "pipeline_class": xFuserPixArtSigmaPipeline,
        "inference_steps": 20,
        "needs_t5": True,
        "t5_subfolder": "text_encoder",
        "dtype": torch.float16,
        "supports_video": False
    },
    "stabilityai/stable-diffusion-3-medium-diffusers": {
        "pipeline_class": xFuserStableDiffusion3Pipeline,
        "inference_steps": 20,
        "needs_t5": True,
        "t5_subfolder": "text_encoder_3",
        "dtype": torch.float16,
        "supports_video": False
    },
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers": {
        "pipeline_class": xFuserHunyuanDiTPipeline,
        "inference_steps": 50,
        "needs_t5": True,
        "t5_subfolder": "text_encoder_2",
        "dtype": torch.float16,
        "supports_video": False
    }
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
    input_config: Any,
    local_rank: int,
    args: DiffusionArgs
) -> tuple[Any, int]:
    """Setup the appropriate pipeline based on model_id."""
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_id: {model_id}")
    
    config = MODEL_CONFIGS[model_id]

    cache_args = {
        "use_teacache": False,
        "use_fbcache": False,
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": input_config.num_inference_steps,
    }
    
    # Handle T5 encoder if needed
    text_encoder_kwargs = {}
    if config["needs_t5"]:
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder=config["t5_subfolder"],
            torch_dtype=config["dtype"]
        )

        model_type = get_model_type_from_id(model_id)
        if model_type == "Flux":
            text_encoder_kwargs["text_encoder_2"] = text_encoder
        elif model_type == "Sd3":
            text_encoder_kwargs["text_encoder_3"] = text_encoder
        elif model_type == "HunyuanDiT":
            text_encoder_kwargs["text_encoder_2"] = text_encoder
        elif model_type == "Pixart-sigma":
            text_encoder_kwargs["text_encoder"] = text_encoder
    
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

    return pipe, config["inference_steps"]


def get_inference_kwargs(
    request: DiffusionRequest,
    model_id: str,
    guidance_scale: float,
    save_images: bool = True
) -> Any:
    inference_kwargs = {
        "prompt": request.prompts,
        "height": request.height,
        "width": request.width,
        "num_inference_steps": request.inference_steps,
        "guidance_scale": guidance_scale,
        "output_type": "pil" if save_images else "latent",
        "generator": torch.Generator(device="cuda").manual_seed(request.seed),
    }
    
    model_type = get_model_type_from_id(model_id)
    if model_type == "Flux":
        inference_kwargs["max_sequence_length"] = 256
    elif model_type in ["Pixart-alpha", "Pixart-sigma", "HunyuanDiT"]:
        inference_kwargs["use_resolution_binning"] = True
    
    # Add video parameters if present
    if hasattr(request, 'num_frames') and request.num_frames is not None:
        inference_kwargs["num_frames"] = request.num_frames
    if hasattr(request, 'fps') and request.fps is not None:
        inference_kwargs["fps"] = request.fps
    
    return inference_kwargs


def save_generated_images(
    pipe: Any,
    output: Any,
    request: Any,
    args: DiffusionArgs,
    output_dir: Path
):
    if args.save_images:
        # output_dir.mkdir(parents=True, exist_ok=True)
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        num_prompts = len(request.prompts)
        dp_batch_size = (num_prompts + num_dp_groups - 1) // num_dp_groups
        
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                if image_rank < num_prompts:
                     prompt_text = request.prompts[image_rank]
                     prompt_snippet = prompt_text[:50]
                     # Remove/replace special characters for safe filename
                     safe_snippet = "".join(c for c in prompt_snippet if c.isalnum() or c in (' ', '-', '_')).rstrip()
                     safe_snippet = safe_snippet.replace(' ', '_')[:30]
                     filename = f"i{image_rank:04d}_{safe_snippet}.png"
                     image_path = output_dir / filename
                     image.save(image_path)
                     logger.info(f"Saved image {image_rank} to {image_path}")


def save_results(
    args: DiffusionArgs,
    request: Any,
    output: Any,
    benchmark_duration: float,
    energy_result: Any = None,
    image_paths: list[str] = None,
    local_rank: int = 0
) -> None:
    if local_rank != 0:
        return
        
    # Calculate metrics
    num_images = len(request.prompts)
    throughput = num_images / benchmark_duration if benchmark_duration > 0 else 0.0
    avg_time_per_image = benchmark_duration / num_images if num_images > 0 else 0.0
    
    # Prepare results dictionary
    result_json: dict[str, Any] = {}
    
    # Setup information
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["model_id"] = args.workload.model_id
    result_json["batch_size"] = args.workload.batch_size
    
    # Generation parameters
    result_json["height"] = request.height
    result_json["width"] = request.width 
    result_json["inference_steps"] = request.inference_steps
    result_json["seed"] = request.seed
    
    # Performance metrics  
    result_json["total_images"] = num_images
    result_json["total_time"] = benchmark_duration
    result_json["throughput_images_per_sec"] = throughput
    result_json["avg_time_per_image"] = avg_time_per_image
    
    # Energy metrics
    if energy_result:
        result_json["total_energy"] = energy_result.total_energy
        result_json["energy_per_image"] = energy_result.total_energy / num_images if num_images > 0 else 0.0
        result_json["energy_measurement"] = asdict(energy_result)
    
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
    
    if energy_result:
        logger.info("%-40s: %.2f", "Total Energy (J)", energy_result.total_energy)
        logger.info("%-40s: %.2f", "Energy per Image (J)", energy_result.total_energy / num_images)


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

    # Load input data
    request = args.workload.load_requests()
    
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
        prompt=request.prompts,
        use_torch_compile=args.workload.use_torch_compile,
    )
    engine_config, input_config = xfuser_args.create_config()

    world_group = get_world_group()
    local_rank = world_group.local_rank
    world_size = world_group.world_size

    # Setup xFuser pipeline
    logger.info(f"Setting up xFuser pipeline")
    pipe, default_inference_steps = setup_pipeline(
        args.workload.model_id, engine_config, input_config, local_rank, args
    )
    inference_kwargs = get_inference_kwargs(request, args.workload.model_id, input_config.guidance_scale, args.save_images)

    logger.info(f"Preparing xFuser pipeline")
    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    logger.info(f"Start running diffusion benchmark")
    torch.cuda.synchronize()
    torch.distributed.barrier()
    benchmark_start_time = time.perf_counter()

    if zeus_monitor:
        zeus_monitor.begin_window("generation")
    
    output = pipe(**inference_kwargs)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    if zeus_monitor:
        energy_result = zeus_monitor.end_window("generation")
    
    benchmark_duration = time.perf_counter() - benchmark_start_time
    logger.info(f"End running diffusion benchmark, duration: {benchmark_duration:.2f}s")

    # Save generated images
    output_dir = args.workload.to_path(of="image_outputs")
    save_generated_images(
        pipe=pipe,
        output=output,
        request=request,
        args=args,
        output_dir=output_dir
    )

    if local_rank == 0:
        save_results(
            args=args,
            request=request,
            output=output,
            benchmark_duration=benchmark_duration,
            energy_result=energy_result,
        )
    
    get_runtime_state().destroy_distributed_env()


# TODO: download the model if not exists
# TODO: handle the default configs of different models
# TODO: move SP degree to the workload config
# TODO: handle server log
if __name__ == "__main__":
    args = tyro.cli(DiffusionArgs[TextToImage | TextToVideo])

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

    try:
        main(args)
    except Exception as e:
        logger.exception("An error occurred during the benchmark: %s", e)
        raise 