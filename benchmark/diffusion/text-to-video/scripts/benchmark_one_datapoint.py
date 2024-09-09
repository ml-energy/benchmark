from __future__ import annotations

import json
import argparse
from typing import Any
from pprint import pprint
from pathlib import Path
from contextlib import suppress
from dataclasses import dataclass, field, asdict

import torch
import pynvml
import numpy as np
from PIL import Image
from transformers.trainer_utils import set_seed
from diffusers import (
    ModelMixin,  # type: ignore
    DiffusionPipeline,  # type: ignore
    AnimateDiffPipeline,  # type: ignore
    DDIMScheduler,  # type: ignore
    MotionAdapter,  # type: ignore
) 
from diffusers.utils import export_to_gif  # pyright: reportPrivateImportUsage=false
from zeus.monitor import ZeusMonitor

# Disable torch gradients globally
torch.set_grad_enabled(False)


@dataclass
class Results:
    model: str
    num_parameters: dict[str, int]
    gpu_model: str
    num_inference_steps: int
    num_frames: int
    power_limit: int
    batch_size: int
    num_prompts: int
    total_runtime: float = 0.0
    total_energy: float = 0.0
    average_batch_latency: float = 0.0
    average_generations_per_second: float = 0.0
    average_batch_energy: float = 0.0
    average_power_consumption: float = 0.0
    peak_memory: float = 0.0
    results: list[Result] = field(default_factory=list, repr=False)


@dataclass
class ResultIntermediateBatched:
    batch_latency: float = 0.0
    batch_energy: float = 0.0
    prompts: list[str] = field(default_factory=list)
    frames: np.ndarray | list[list[Image.Image]] = np.empty(0)


@dataclass
class Result:
    batch_latency: float
    sample_energy: float
    prompt: str
    video_path: str | None


def get_pipeline(model_id: str):
    """Instantiate a Diffusers pipeline from a modes's HuggingFace Hub ID."""
    # Load args to give to `from_pretrained` from the model's kwargs.json file
    kwargs = build_kwargs(model_id)

    # Hack for AnimateDiff
    if "animatediff" in model_id:
        adapter = MotionAdapter.from_pretrained(model_id, **kwargs)
        sd_model_id = "emilianJR/epiCRealism"
        sd_kwargs = build_kwargs(sd_model_id)
        pipeline = AnimateDiffPipeline.from_pretrained(sd_model_id, motion_adapter=adapter, **sd_kwargs)
        scheduler = DDIMScheduler.from_pretrained(
            sd_model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipeline.scheduler = scheduler
        pipeline = pipeline.to("cuda:0")
        print("\nInstantiated AnimateDiff pipeline:\n", pipeline)
    else:
        pipeline = DiffusionPipeline.from_pretrained(model_id, **kwargs).to("cuda:0")
        print("\nInstantiated pipeline via DiffusionPipeline:\n", pipeline)

    return pipeline


def build_kwargs(model_id: str) -> dict:
    """Build the kwargs to pass to the model's `from_pretrained` method."""
    kwargs = json.load(open(f"models/{model_id}/kwargs.json"))
    with suppress(KeyError):
        kwargs["torch_dtype"] = eval(kwargs["torch_dtype"])

    # Add additional args
    kwargs["safety_checker"] = None
    kwargs["revision"] = open(f"models/{model_id}/revision.txt").read().strip()

    return kwargs


def load_text_prompts(
    path: str,
    batch_size: int,
    num_batches: int | None = None,
) -> tuple[int, list[list[str]]]:
    """Load the dataset to feed the model and return it as a list of batches of prompts.

    Depending on the batch size, the final batch may not be full. The final batch
    is dropped in that case. If `num_batches` is not None, only that many batches
    is returned. If `num_batches` is None, all batches are returned.

    Returns:
        Total number of prompts and a list of batches of prompts.
    """
    dataset = json.load(open(path))["caption"]
    if num_batches is not None:
        if len(dataset) < num_batches * batch_size:
            raise ValueError("Dataset is too small for the given number of batches.")
        dataset = dataset[:num_batches * batch_size]
    batched = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]
    if len(batched[-1]) < batch_size:
        batched.pop()
    return len(batched) * batch_size, batched


def count_parameters(pipeline) -> dict[str, int]:
    """Count the number of parameters in the given pipeline."""
    num_params = {}
    for name, attr in vars(pipeline).items():
        if isinstance(attr, ModelMixin):
            num_params[name] = attr.num_parameters(only_trainable=False, exclude_embeddings=True)
        elif isinstance(attr, torch.nn.Module):
            num_params[name] = sum(p.numel() for p in attr.parameters())
    return num_params


def benchmark(args: argparse.Namespace) -> None:
    if args.model.startswith("models/"):
        args.model = args.model[len("models/") :]
    if args.model.endswith("/"):
        args.model = args.model[:-1]

    set_seed(args.seed)

    results_dir = Path(args.result_root) / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    benchmark_name = str(results_dir / f"bs{args.batch_size}+pl{args.power_limit}")
    video_dir = results_dir / f"bs{args.batch_size}+pl{args.power_limit}+generated"
    video_dir.mkdir(exist_ok=True)

    arg_out_filename = f"{benchmark_name}+args.json"
    with open(arg_out_filename, "w") as f:
        f.write(json.dumps(vars(args), indent=2))
    print(args)
    print("Benchmark args written to", arg_out_filename)

    zeus_monitor = ZeusMonitor()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_model = pynvml.nvmlDeviceGetName(handle)
    pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, args.power_limit * 1000)
    pynvml.nvmlShutdown()

    num_prompts, batched_prompts = load_text_prompts(args.dataset_path, args.batch_size, args.num_batches)

    pipeline = get_pipeline(args.model)

    # Warmup
    print("Warming up with two batches...")
    for i in range(2):
        _ = pipeline(
            prompt=batched_prompts[i],
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
        )

    rng = torch.manual_seed(args.seed)

    intermediates: list[ResultIntermediateBatched] = [
        ResultIntermediateBatched(prompts=batch) for batch in batched_prompts
    ]

    torch.cuda.reset_peak_memory_stats(device="cuda:0")
    zeus_monitor.begin_window("benchmark", sync_cuda=False)

    # Build common parameter dict for all batches
    params: dict[str, Any] = dict(
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        generator=rng,
    )
    if args.height is not None:
        params["height"] = args.height
    if args.width is not None:
        params["width"] = args.width

    for ind, intermediate in enumerate(intermediates):
        print(f"Batch {ind + 1}/{len(intermediates)}")

        params["prompt"] = intermediate.prompts

        zeus_monitor.begin_window("batch", sync_cuda=False)
        frames = pipeline(**params).frames
        batch_measurements = zeus_monitor.end_window("batch", sync_cuda=False)

        intermediate.frames = frames
        intermediate.batch_latency = batch_measurements.time
        intermediate.batch_energy = batch_measurements.total_energy

    measurements = zeus_monitor.end_window("benchmark", sync_cuda=False)
    peak_memory = torch.cuda.max_memory_allocated(device="cuda:0")

    results: list[Result] = []
    ind = 0
    for intermediate in intermediates:
        # Some pipelines just return a giant numpy array for all frames.
        # In that case, scale frames to uint8 [0, 256] and convert to PIL.Image
        if isinstance(intermediate.frames, np.ndarray):
            frames = []
            for batch in intermediate.frames:
                frames.append(
                    [Image.fromarray((frame * 255).astype(np.uint8)) for frame in batch]
                )
            intermediate.frames = frames

        for frames, prompt in zip(intermediate.frames, intermediate.prompts, strict=True):
            if ind % args.save_every == 0:
                video_path = str(video_dir / f"{prompt[:200]}.gif")
                export_to_gif(frames, video_path)
            else:
                video_path = None

            results.append(
                Result(
                    batch_latency=intermediate.batch_latency,
                    sample_energy=intermediate.batch_energy / len(intermediate.prompts),
                    prompt=prompt,
                    video_path=video_path,
                )
            )
            ind += 1

    final_results = Results(
        model=args.model,
        num_parameters=count_parameters(pipeline),
        gpu_model=gpu_model,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        power_limit=args.power_limit,
        batch_size=args.batch_size,
        num_prompts=num_prompts,
        total_runtime=measurements.time,
        total_energy=measurements.total_energy,
        average_batch_latency=measurements.time / len(batched_prompts),
        average_generations_per_second=num_prompts / measurements.time,
        average_batch_energy=measurements.total_energy / len(batched_prompts),
        average_power_consumption=measurements.total_energy / measurements.time,
        peak_memory=peak_memory,
        results=results,
    )

    with open(f"{benchmark_name}+results.json", "w") as f:
        f.write(json.dumps(asdict(final_results), indent=2))
    print("Benchmark results written to", f"{benchmark_name}+results.json")

    print("Benchmark results:")
    pprint(final_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to benchmark.")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset to use.")
    parser.add_argument("--result-root", type=str, help="The root directory to save results to.")
    parser.add_argument("--batch-size", type=int, default=1, help="The size of each batch of prompts.")
    parser.add_argument("--power-limit", type=int, default=300, help="The power limit to set for the GPU in Watts.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="The number of denoising steps.")
    parser.add_argument("--num-frames", type=int, default=16, help="The number of frames to generate.")
    parser.add_argument("--height", type=int, help="Height of the generated video.")
    parser.add_argument("--width", type=int, help="Width of the generated video.")
    parser.add_argument("--num-batches", type=int, default=None, help="The number of batches to use from the dataset.")
    parser.add_argument("--save-every", type=int, default=10, help="Save images to file every N prompts.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use for the RNG.")
    parser.add_argument("--huggingface-token", type=str, help="The HuggingFace token to use.")
    args = parser.parse_args()

    benchmark(args)
