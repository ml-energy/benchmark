from __future__ import annotations

import os
import json
import argparse
from pprint import pprint
from pathlib import Path
from contextlib import suppress
from dataclasses import dataclass, field, asdict

import torch
import pynvml
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset
from transformers.trainer_utils import set_seed
from transformers import CLIPModel, CLIPProcessor
from diffusers import (
    ModelMixin,  # type: ignore
    AutoPipelineForText2Image,  # type: ignore
    DiffusionPipeline,  # type: ignore
    StableCascadeCombinedPipeline,  # type: ignore
) 
from zeus.monitor import ZeusMonitor

# Disable torch gradients globally
torch.set_grad_enabled(False)


CLIP = "openai/clip-vit-large-patch14"


@dataclass
class Results:
    model: str
    num_parameters: dict[str, int]
    gpu_model: str
    num_inference_steps: int
    power_limit: int
    batch_size: int
    num_prompts: int
    average_clip_score: float = 0.0
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
    batch_latency: float = 0.0
    batch_energy: float = 0.0
    prompts: list[str] = field(default_factory=list)
    images: np.ndarray = np.empty(0)


@dataclass
class Result:
    batch_latency: float
    sample_energy: float
    prompt: str
    image_path: str | None
    clip_score: float


def get_pipeline(model_id: str):
    """Instantiate a Diffusers pipeline from a modes's HuggingFace Hub ID."""
    # Load args to give to `from_pretrained` from the model's kwargs.json file
    kwargs = json.load(open(f"models/{model_id}/kwargs.json"))
    with suppress(KeyError):
        kwargs["torch_dtype"] = eval(kwargs["torch_dtype"])

    # Add additional args
    kwargs["safety_checker"] = None
    kwargs["revision"] = open(f"models/{model_id}/revision.txt").read().strip()

    # Hack for stable-cascade, which defaults to only a part of the model.
    if model_id == "stabilityai/stable-cascade":
        pipeline = StableCascadeCombinedPipeline.from_pretrained(model_id, **kwargs).to("cuda:0")
        print("\nInstantiated pipeline via StableCascadeCombinedPipeline:\n", pipeline)
    else:
        try:
            pipeline =  AutoPipelineForText2Image.from_pretrained(model_id, **kwargs).to("cuda:0")
            print("\nInstantiated pipeline via AutoPipelineForText2Image:\n", pipeline)
        except ValueError:
            pipeline = DiffusionPipeline.from_pretrained(model_id, **kwargs).to("cuda:0")
            print("\nInstantiated pipeline via DiffusionPipeline:\n", pipeline)

    return pipeline


def load_partiprompts(
    batch_size: int,
    seed: int,
    num_batches: int | None = None,
) -> tuple[int, list[list[str]]]:
    """Load the parti-prompts dataset and return it as a list of batches of prompts.

    Depending on the batch size, the final batch may not be full. The final batch
    is dropped in that case. If `num_batches` is not None, only that many batches
    is returned. If `num_batches` is None, all batches are returned.

    Returns:
        Total number of prompts and a list of batches of prompts.
    """
    dataset = load_dataset("nateraw/parti-prompts", split="train").shuffle(seed=seed)
    assert isinstance(dataset, Dataset)
    if num_batches is not None:
        dataset = dataset.select(range(min(num_batches * batch_size, len(dataset))))
    prompts: list[str] = dataset["Prompt"]
    batched = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    if len(batched[-1]) < batch_size:
        batched.pop()
    return len(batched) * batch_size, batched


def calculate_clip_score(
    model: CLIPModel,
    processor: CLIPProcessor,
    images_np: np.ndarray,
    text: list[str],
) -> torch.Tensor:
    """Calculate the CLIP score for each image and prompt pair.

    `images_np` is assumed to be already scaled to [0, 255] and in uint8 format.

    Returns:
        The clip score of each image and prompt as a list of floats.
        Tensor shape is (batch size,).
    """
    model = model.to("cuda:0")

    images = list(torch.from_numpy(images_np).permute(0, 3, 1, 2))
    assert len(images) == len(text)
    
    processed_input = processor(text=text, images=images, return_tensors="pt", padding=True)
    img_features = model.get_image_features(processed_input["pixel_values"].to("cuda:0"))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    max_position_embeddings = model.config.text_config.max_position_embeddings
    if processed_input["attention_mask"].shape[-1] > max_position_embeddings:
        print(
            f"Input attention mask is larger than max_position_embeddings. "
            f"Truncating the attention mask to {max_position_embeddings}."
        )
        processed_input["attention_mask"] = processed_input["attention_mask"][..., :max_position_embeddings]
        processed_input["input_ids"] = processed_input["input_ids"][..., :max_position_embeddings]

    txt_features = model.get_text_features(
        processed_input["input_ids"].to("cuda:0"), processed_input["attention_mask"].to("cuda:0")
    )
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

    scores = 100 * (img_features * txt_features).sum(axis=-1)
    scores = torch.max(scores, torch.zeros_like(scores))
    return scores


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
    os.environ["HF_TOKEN"] = args.huggingface_token

    if args.model.startswith("models/"):
        args.model = args.model[len("models/") :]
    if args.model.endswith("/"):
        args.model = args.model[:-1]

    set_seed(args.seed)

    results_dir = Path(args.result_root) / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    benchmark_name = str(results_dir / f"bs{args.batch_size}+pl{args.power_limit}")
    image_dir = results_dir / f"bs{args.batch_size}+pl{args.power_limit}+generated"
    image_dir.mkdir(exist_ok=True)

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

    num_prompts, batched_prompts = load_partiprompts(args.batch_size, args.seed, args.num_batches)

    pipeline = get_pipeline(args.model)

    # Warmup
    print("Warming up with five batches...")
    for i in range(5):
        _ = pipeline(
            batched_prompts[i],
            num_inference_steps=args.num_inference_steps,
            output_type="np",
        )

    rng = torch.manual_seed(args.seed)

    images = []
    intermediates: list[ResultIntermediateBatched] = [
        ResultIntermediateBatched(prompts=batch) for batch in batched_prompts
    ]

    torch.cuda.reset_peak_memory_stats(device="cuda:0")
    zeus_monitor.begin_window("benchmark", sync_cuda=False)

    for ind, intermediate in enumerate(intermediates):
        print(f"Batch {ind + 1}/{len(intermediates)}")
        zeus_monitor.begin_window("batch", sync_cuda=False)
        images = pipeline(
            intermediate.prompts,
            generator=rng,
            num_inference_steps=args.num_inference_steps,
            output_type="np",
        ).images
        batch_measurements = zeus_monitor.end_window("batch", sync_cuda=False)

        intermediate.images = images
        intermediate.batch_latency = batch_measurements.time
        intermediate.batch_energy = batch_measurements.total_energy

    measurements = zeus_monitor.end_window("benchmark", sync_cuda=False)
    peak_memory = torch.cuda.max_memory_allocated(device="cuda:0")

    # Scale images to [0, 256] and convert to uint8
    for intermediate in intermediates:
        intermediate.images = (intermediate.images * 255).astype("uint8")

    # Compute the CLIP score for each image and prompt pair.
    # Code was mostly inspired from torchmetrics.multimodal.clip_score, but
    # adapted here to calculate the CLIP score for each image and prompt pair.
    clip_model: CLIPModel = CLIPModel.from_pretrained(CLIP).cuda()  # type: ignore
    clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(CLIP)  # type: ignore
    batch_clip_scores = []
    for intermediate in intermediates:
        clip_score = calculate_clip_score(
            clip_model,
            clip_processor,
            intermediate.images,
            intermediate.prompts,
        )
        batch_clip_scores.append(clip_score.tolist())

    results: list[Result] = []
    ind = 0
    for intermediate, batch_clip_score in zip(intermediates, batch_clip_scores, strict=True):
        for image, prompt, clip_score in zip(
            intermediate.images,
            intermediate.prompts,
            batch_clip_score,
            strict=True,
        ):
            if ind % args.image_save_every == 0:
                image_path = str(image_dir / f"{prompt}.png")
                Image.fromarray(image).save(image_path)
            else:
                image_path = None

            results.append(
                Result(
                    batch_latency=intermediate.batch_latency,
                    sample_energy=intermediate.batch_energy / len(intermediate.prompts),
                    prompt=prompt,
                    image_path=image_path,
                    clip_score=clip_score,
                )
            )
            ind += 1

    final_results = Results(
        model=args.model,
        num_parameters=count_parameters(pipeline),
        gpu_model=gpu_model,
        num_inference_steps=args.num_inference_steps,
        power_limit=args.power_limit,
        batch_size=args.batch_size,
        num_prompts=num_prompts,
        average_clip_score=sum(r.clip_score for r in results) / len(results),
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
    parser.add_argument("--result-root", type=str, help="The root directory to save results to.")
    parser.add_argument("--batch-size", type=int, default=1, help="The size of each batch of prompts.")
    parser.add_argument("--power-limit", type=int, default=300, help="The power limit to set for the GPU in Watts.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="The number of denoising steps.")
    parser.add_argument("--num-batches", type=int, default=None, help="The number of batches to use from the dataset.")
    parser.add_argument("--image-save-every", type=int, default=10, help="Save images to file every N prompts.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use for the RNG.")
    parser.add_argument("--huggingface-token", type=str, help="The HuggingFace token to use.")
    args = parser.parse_args()

    benchmark(args)
