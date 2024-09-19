from __future__ import annotations

import json
import inspect
import argparse
from pprint import pprint
from pathlib import Path
from contextlib import suppress
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import pynvml
import numpy as np
from PIL import Image
from transformers.trainer_utils import set_seed
from diffusers import ModelMixin, DiffusionPipeline  # type: ignore
from diffusers.utils import load_image, export_to_gif  # pyright: reportPrivateImportUsage=false
from zeus.monitor import ZeusMonitor

# Disable torch gradients globally
torch.set_grad_enabled(False)


@dataclass
class Results:
    model: str
    num_parameters: dict[str, int]
    gpu_model: str
    power_limit: int
    batch_size: int
    num_inference_steps: int
    num_frames: int
    num_prompts: int
    total_runtime: float = 0.0
    total_energy: float = 0.0
    average_batch_latency: float = 0.0
    average_images_per_second: float = 0.0
    average_batch_energy: float = 0.0
    average_power_consumption: float = 0.0
    peak_memory: float = 0.0
    results: list[Result] = field(default_factory=list, repr=False)


@dataclass
class ResultIntermediateBatched:
    prompts: list[str]
    images: list[Image.Image]
    batch_latency: float = 0.0
    batch_energy: float = 0.0
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
    kwargs = json.load(open(f"models/{model_id}/kwargs.json"))
    with suppress(KeyError):
        kwargs["torch_dtype"] = eval(kwargs["torch_dtype"])

    # Add additional args
    kwargs["safety_checker"] = None
    kwargs["revision"] = open(f"models/{model_id}/revision.txt").read().strip()

    pipeline = DiffusionPipeline.from_pretrained(model_id, **kwargs).to("cuda:0")
    print("\nInstantiated pipeline via DiffusionPipeline:\n", pipeline)

    return pipeline


def load_text_image_prompts(
    path: str,
    batch_size: int,
    num_batches: int | None = None,
    image_resize: tuple[int, int] | None = None,
) -> tuple[int, list[tuple[list[str], list[Image.Image]]]]:
    """Load the dataset to feed the model and return it as a list of batches of prompts.

    Depending on the batch size, the final batch may not be full. The final batch
    is dropped in that case. If `num_batches` is not None, only that many batches
    is returned. If `num_batches` is None, all batches are returned.

    Returns:
        Total number of prompts and a list of batches of prompts.
    """
    dataset = json.load(open(path))
    assert len(dataset["caption"]) == len(dataset["video_id"])

    dataset["caption"] *= 10
    dataset["video_id"] *= 10

    if num_batches is not None:
        if len(dataset["caption"]) < num_batches * batch_size:
            raise ValueError("Not enough data for the requested number of batches.")
        dataset["caption"] = dataset["caption"][: num_batches * batch_size]
        dataset["video_id"] = dataset["video_id"][: num_batches * batch_size]

    image_path = Path(path).parent / "first_frame"
    dataset["first_frame"] = [
        load_image(str(image_path / f"{video_id}.jpg")) for video_id in dataset["video_id"]
    ]
    if image_resize is not None:
        dataset["first_frame"] = [image.resize(image_resize) for image in dataset["first_frame"]]

    batched = [
        (dataset["caption"][i : i + batch_size], dataset["first_frame"][i : i + batch_size])
        for i in range(0, len(dataset["caption"]), batch_size)
    ]
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
    benchmark_name = str(results_dir / f"bs{args.batch_size}+pl{args.power_limit}+steps{args.num_inference_steps}")
    video_dir = results_dir / f"bs{args.batch_size}+pl{args.power_limit}+steps{args.num_inference_steps}+generated"
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
    # pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
    # pynvml.nvmlDeviceSetPowerManagementLimit(handle, args.power_limit * 1000)
    pynvml.nvmlShutdown()

    num_prompts, batched_prompts = load_text_image_prompts(
        args.dataset_path,
        args.batch_size,
        args.num_batches,
        (args.width, args.height),
    )

    pipeline = get_pipeline(args.model)

    # Warmup
    print("Warming up with two batches...")
    for i in range(2):
        params: dict[str, Any] = dict(
            image=batched_prompts[i][1],
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
        )
        if args.add_text_prompt:
            params["prompt"] = batched_prompts[i][0]

        _ = pipeline(**params)

    rng = torch.manual_seed(args.seed)

    # Some models require a text prompt alongside the image (e.g., I2VGen-XL)
    # In that case, `prompts` will not be passed to the model.
    intermediates: list[ResultIntermediateBatched] = [
        ResultIntermediateBatched(prompts=text, images=image) for text, image in batched_prompts
    ]

    # Different pipelines use different names for the FPS parameter
    gen_signature= inspect.signature(pipeline.__call__)
    fps_param_name_candidates = list(filter(lambda x: "fps" in x, gen_signature.parameters))
    if not fps_param_name_candidates:
        raise ValueError("No parameter with 'fps' in its name found in the pipeline's signature.")
    if len(fps_param_name_candidates) > 1:
        raise ValueError("Multiple parameters with 'fps' in their name found in the pipeline's signature.")
    fps_param_name = fps_param_name_candidates[0]

    torch.cuda.reset_peak_memory_stats(device="cuda:0")
    zeus_monitor.begin_window("benchmark", sync_execution=False)

    # Build common parameter dict for all batches
    params: dict[str, Any] = dict(
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        generator=rng,
    )
    params[fps_param_name] = args.fps
    if args.height is not None:
        params["height"] = args.height
    if args.width is not None:
        params["width"] = args.width

    for ind, intermediate in enumerate(intermediates):
        print(f"Batch {ind + 1}/{len(intermediates)}")

        params["image"] = intermediate.images
        if args.add_text_prompt:
            params["prompt"] = intermediate.prompts

        zeus_monitor.begin_window("batch", sync_execution=False)
        frames = pipeline(**params).frames
        batch_measurements = zeus_monitor.end_window("batch", sync_execution=False)

        intermediate.frames = frames
        intermediate.batch_latency = batch_measurements.time
        intermediate.batch_energy = batch_measurements.total_energy

    measurements = zeus_monitor.end_window("benchmark", sync_execution=False)
    peak_memory = torch.cuda.max_memory_allocated(device="cuda:0")

    results: list[Result] = []
    ind = 0
    for intermediate in intermediates:
        # Some pipelines just return a giant numpy array for all frames.
        # In that case, scale frames to uint8 [0, 256] and convert to PIL.Image
        if isinstance(intermediate.frames, np.ndarray):
            frames = []
            for video in intermediate.frames:
                frames.append(
                    [Image.fromarray((frame * 255).astype(np.uint8)) for frame in video]
                )
            intermediate.frames = frames

        for frames, prompt in zip(intermediate.frames, intermediate.prompts, strict=True):
            if ind % args.save_every == 0:
                video_path = str(video_dir / f"{prompt[:200]}.gif")
                export_to_gif(frames, video_path, fps=args.fps)
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
        power_limit=args.power_limit,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        num_prompts=num_prompts,
        total_runtime=measurements.time,
        total_energy=measurements.total_energy,
        average_batch_latency=measurements.time / len(batched_prompts),
        average_images_per_second=num_prompts / measurements.time,
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
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset to use.")
    parser.add_argument("--add-text-prompt", action="store_true", help="Input text prompt alongside image.")
    parser.add_argument("--result-root", type=str, help="The root directory to save results to.")
    parser.add_argument("--batch-size", type=int, default=1, help="The size of each batch of prompts.")
    parser.add_argument("--power-limit", type=int, default=300, help="The power limit to set for the GPU in Watts.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="The number of denoising steps.")
    parser.add_argument("--num-frames", type=int, default=1, help="The number of frames to generate.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for micro-conditioning.")
    parser.add_argument("--height", type=int, required=True, help="Height of the generated video.")
    parser.add_argument("--width", type=int, required=True, help="Width of the generated video.")
    parser.add_argument("--num-batches", type=int, default=None, help="The number of batches to use from the dataset.")
    parser.add_argument("--save-every", type=int, default=10, help="Save generations to file every N prompts.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use for the RNG.")
    parser.add_argument("--huggingface-token", type=str, help="The HuggingFace token to use.")
    args = parser.parse_args()

    benchmark(args)
