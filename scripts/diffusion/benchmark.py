import gc
import time

from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import numpy as np
from PIL import Image
import torch
from transformers.trainer_utils import set_seed
import tyro
from zeus.monitor import ZeusMonitor

from utils import get_logger, CsvHandler
from metrics import load_prompts, calculate_clip_score

# default parameters
DEVICE = "cuda:0"
WEIGHT_DTYPE = torch.float16
SEED = 0
OUTPUT_FILE = "results.csv"
OUTPUT_IMAGES = "images/"


def get_pipeline(model, device=DEVICE, weight_dtype=WEIGHT_DTYPE):
    try:
        return AutoPipelineForText2Image.from_pretrained(
            model, torch_dtype=weight_dtype, safety_checker=None
        ).to(device)
    except ValueError:
        return DiffusionPipeline.from_pretrained(
            model, torch_dtype=weight_dtype, safety_checker=None
        ).to(device)


def gpu_warmup(pipeline):
    """Warm up the GPU by running the given pipeline for 10 secs."""
    logger = get_logger()
    logger.info("Warming up GPU")
    generator = torch.manual_seed(2)
    timeout_start = time.time()
    prompts, _ = load_prompts(1, 1)
    while time.time() < timeout_start + 10:
        _ = pipeline(
            prompts, num_images_per_prompt=10, generator=generator, output_type="numpy"
        ).images
    logger.info("Finished warming up GPU")


def benchmark(
    model: str,
    benchmark_size: int = 0,
    batch_size: int = 1,
    result_file: str = OUTPUT_FILE,
    images_path: str = OUTPUT_IMAGES,
    device: str = DEVICE,
    seed: int = SEED,
    weight_dtype: torch.dtype = WEIGHT_DTYPE,
    write_header: bool = False,
    warmup: bool = False,
    settings: dict = {},
) -> None:
    """Benchmarks given model with a set of parameters.

    Args:
        model: The name of the model to benchmark, as shown on HuggingFace.
        benchmark_size: The number of prompts to benchmark on. If 0, benchmarks
          the entire parti-prompts dataset.
        batch_size: The size of each batch of prompts. When benchmarking, the
          prompts are split into batches of this size, and prompts are fed into
          the model in batches.
        result_file: The path to the output csv file.
        images_path: The path to the output images directory.
        device: The device to run the benchmark on.
        seed: The seed to use for the RNG.
        weight_dtype: The weight dtype to use for the model.
        write_header: Whether to write the header row to the output csv file,
          recommended to be True for the first run.
        warmup: Whether to warm up the GPU before running the benchmark,
          recommended to be True for the first run of a model.
        settings: Any additional settings to pass to the pipeline, supports
          any keyword parameters accepted by the model chosen. See HuggingFace
          documentation on particular models for more details.
    """
    logger = get_logger()
    logger.info("Running benchmark for model: " + model)

    csv_handler = CsvHandler(result_file)
    if write_header:
        csv_handler.write_header(
            [
                "model",
                "GPU",
                "num_prompts",
                "batch_size",
                "clip_score",
                "average_batch_latency(s)",
                "throughput(image/s)",
                "avg_energy(J)",
                "peak_memory(GB)",
            ]
        )

    set_seed(seed)
    prompts, batched_prompts = load_prompts(benchmark_size, batch_size)
    logger.info("Loaded prompts")

    generator = torch.manual_seed(seed)
    torch.cuda.set_device(device)
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    pipeline = get_pipeline(model, device=device, weight_dtype=weight_dtype)

    if warmup:
        gpu_warmup(pipeline)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device=device)

    monitor.begin_window("generate")
    images = []
    for batch in batched_prompts:
        image = pipeline(
            batch, generator=generator, output_type="np", **settings
        ).images
        images.append(image)
    images = np.concatenate(images)
    result_monitor = monitor.end_window("generate")

    peak_memory = torch.cuda.max_memory_allocated(device=device)

    for saved_image, saved_prompt in zip(images[::10], prompts[::10]):
        saved_image = (saved_image * 255).astype(np.uint8)
        Image.fromarray(saved_image).save(images_path + saved_prompt + ".png")

    clip_score = calculate_clip_score(images, prompts)

    result = {
        "model": model,
        "GPU": torch.cuda.get_device_name(device),
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "clip_score": clip_score,
        "avg_batch_latency": result_monitor.time / (benchmark_size / batch_size),
        "throughput": benchmark_size / result_monitor.time,
        "avg_energy": result_monitor.total_energy / benchmark_size,
        "peak_memory": peak_memory,
    }

    logger.info("Results for model " + model + ":")
    logger.info(result)

    csv_handler.write_results(result)

    logger.info("Finished benchmarking for " + model)


if __name__ == "__main__":
    tyro.cli(benchmark)
