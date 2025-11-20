"""Workload configurations for diffusion model benchmarks.

Similar to LLM workloads, this defines specific cases for benchmarking diffusion models.
"""

from __future__ import annotations

import os
import json
import logging
import random
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self, Optional

from datasets import load_dataset
from pydantic import BaseModel, model_validator

from mlenergy.diffusion.dataset import DiffusionRequest, OpenPreferenceDataset, EvalCrafterDataset
from mlenergy.constants import DEFAULT_SEED

logger = logging.getLogger(__name__)


# Model configurations - map model_id to pipeline and settings
MODEL_CONFIGS = {
    # https://huggingface.co/black-forest-labs/FLUX.1-dev
    "black-forest-labs/FLUX.1-dev": {
        "inference_steps": 50,
        "height": 1024,
        "width": 1024,
        "num_frames": None,
        "fps": None,
    },
    # https://huggingface.co/docs/diffusers/en/api/pipelines/pixart_sigma
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": {
        "inference_steps": 20,
        "height": 1024,
        "width": 1024,
        "num_frames": None,
        "fps": None,
    },
    # https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
    "stabilityai/stable-diffusion-3-medium-diffusers": {
        "inference_steps": 28,
        "height": 1024, 
        "width": 1024,
        "num_frames": None,
        "fps": None,
    },
    # https://huggingface.co/docs/diffusers/en/api/pipelines/hunyuandit
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers": {
        "inference_steps": 50,
        "height": 1024,
        "width": 1024,
        "num_frames": None,
        "fps": None,
    },
    # https://github.com/NVlabs/Sana
    "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers": {
        "inference_steps": 20,
        "height": 1024,
        "width": 1024,
        "num_frames": None,
        "fps": None,
    },
    # https://huggingface.co/zai-org/CogVideoX1.5-5B
    "zai-org/CogVideoX1.5-5B": {
        "inference_steps": 50,
        "height": 768,
        "width": 1360,
        "num_frames": 81,
        "fps": 8,
    },
    # https://huggingface.co/BestWishYsh/ConsisID-preview
    "BestWishYsh/ConsisID-preview": {
        "inference_steps": 50,
        "height": 480, # Some SP degree may fail because not divisible
        "width": 720,
        "num_frames": 49,
        "fps": 8,
    },
    # https://huggingface.co/docs/diffusers/main/en/api/pipelines/latte
    "maxin-cn/Latte-1": {
        "inference_steps": 50,
        "height": 512,
        "width": 512,
        "num_frames": 16,
        "fps": 8,
    },

}


def get_model_defaults(model_id: str) -> dict[str, Any]:
    """Get default configuration for a specific model."""
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_id: {model_id}")

    config = MODEL_CONFIGS[model_id]
    return {
        "height": config["height"],
        "width": config["width"],
        "inference_steps": config["inference_steps"],
        "num_frames": config["num_frames"],
        "fps": config["fps"],
    }


class DiffusionWorkloadConfig(BaseModel):
    """Base class for diffusion workload configurations.

    A workload configuration defines one specific case or datapoint for benchmarking.
    It should instantiate appropriate datasets lazily and provide methods to sample
    and save requests.

    Attributes:
        base_dir: Base directory where all workload files are stored.
            It should be unique for each workload configuration.
        seed: Random seed for reproducibility.
        model_id: Model identifier (e.g. black-forest-labs/FLUX.1-dev)
        batch_size: Number of requests to sample for the benchmark.
        height: Output image height
        width: Output image width
        inference_steps: Number of inference steps
        use_torch_compile: Whether to use torch.compile optimization
        ulysses_degree: Ulysses attention parallelism degree
        ring_degree: Ring attention parallelism degree
    """

    # Input parameters
    base_dir: Path
    seed: int = DEFAULT_SEED
    model_id: str
    batch_size: int
    
    # Generation parameters - will use model-specific defaults if not specified
    height: Optional[int] = None
    width: Optional[int] = None
    inference_steps: Optional[int] = None
    
    # Parallelism configuration
    ulysses_degree: int = 1
    ring_degree: int = 1

    # Optimization parameters
    use_torch_compile: bool = False

    @model_validator(mode="after")
    def _validate_workload(self) -> Self:
        """Validate the sanity of the workload and apply model-specific defaults."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        
        # Apply model-specific defaults for parameters that weren't specified
        defaults = get_model_defaults(self.model_id)

        if self.height is None:
            self.height = defaults.get("height", 1024)
        if self.width is None:
            self.width = defaults.get("width", 1024)
        if self.inference_steps is None:
            self.inference_steps = defaults.get("inference_steps", 28)

        return self

    def to_path(
        self,
        of: Literal[
            "requests", "results", "driver_log", "server_log", "image_outputs"
        ],
    ) -> Path:
        """Generate a file path based on file type and workload parameters.

        Types of paths:
        - requests: Path to the file where sampled requests are saved.
        - results: Path to the file where results of the benchmark are saved.
        - driver_log: Path to the file where logging outputs from the driver are saved.
        - server_log: Path to the file where logging outputs from the server are saved.
        - image_outputs: Path to the directory where generated images are saved.
        """
        dir = self.base_dir / "+".join(self.to_filename_parts())

        match of:
            case "requests":
                append = "requests.json"
            case "results":
                append = "results.json"
            case "driver_log":
                append = "driver_log.txt"
            case "server_log":
                append = "server_log.txt"
            case "image_outputs":
                append = "image_outputs"
            case _:
                raise ValueError(f"Unknown path type: {of}")

        path = dir / append
        if not path.suffix:  # Directory
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def to_filename_parts(self) -> list[str]:
        """Generate filename parts based on workload parameters."""
        # Clean model_id for filename
        model_name = self.model_id.replace("/", "-").replace(".", "_")
        parts = [
            f"batch-{self.batch_size}",
            f"size-{self.height}x{self.width}",
            f"steps-{self.inference_steps}",
            f"seed-{self.seed}",
            f"uly-{self.ulysses_degree}",
            f"ring-{self.ring_degree}",
            f"tc-{self.use_torch_compile}",
        ]
        return parts

    def load_requests(self, warmup_iters, benchmark_iters) -> list[DiffusionRequest]:
        """Load the requests from the file specified by the configuration.

        If the file does not exist, it will call `sample` to sample new requests.
        
        Args:
            warmup_iters: Number of warmup iterations
            benchmark_iters: Number of benchmark iterations
            
        Returns:
            List of DiffusionRequest objects, one for each iteration
        """
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        total_iters = warmup_iters + benchmark_iters

        # Determine a shared cache directory like run/diffusion/text-to-image or text-to-video
        def find_category_dir(base: Path) -> Path | None:
            for p in [base] + list(base.parents):
                if p.name in {"text-to-image", "text-to-video"}:
                    return p
            return None

        category_dir = find_category_dir(self.base_dir)
        model_request_path = self.to_path(of="requests")

        shared_requests_path = None
        if category_dir is not None:
            total_prompts = total_iters * self.batch_size
            shared_requests_path = category_dir / f"requests-totalprompts-{total_prompts}-seed-{self.seed}.json"
            shared_requests_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer loading from shared cache if available, otherwise fall back to model-specific cache
        if shared_requests_path is not None and shared_requests_path.exists():
            logger.info(f"Loading cached requests from {shared_requests_path}")
            with open(shared_requests_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                requests = [DiffusionRequest(**req_data) for req_data in data]
            else:
                requests = [DiffusionRequest(**data)]
            return requests
        if model_request_path.exists():
            logger.info(f"Loading cached requests from {model_request_path}")
            with open(model_request_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                requests = [DiffusionRequest(**req_data) for req_data in data]
            else:
                requests = [DiffusionRequest(**data)]
            return requests

        # Neither cache exists; create new requests
        logger.info(f"Creating new requests for {self.model_id}")
        requests = self.sample(total_iters)
        if local_rank == 0:
            if shared_requests_path is not None:
                with open(shared_requests_path, 'w', encoding='utf-8') as f:
                    json.dump([req.model_dump() for req in requests], f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(requests)} requests to {shared_requests_path}")
            with open(model_request_path, 'w', encoding='utf-8') as f:
                json.dump([req.model_dump() for req in requests], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(requests)} requests to {model_request_path}")
        return requests

    def sample(self, num_requests: int) -> list[DiffusionRequest]:
        """Sample requests from the dataset.
        
        This method should be implemented by subclasses to specify dataset-specific sampling logic.
        
        Args:
            num_requests: Total number of DiffusionRequest objects to create
            
        Returns:
            List of DiffusionRequest objects, each potentially with different prompts/seeds
        """
        raise NotImplementedError("Subclasses must implement the sample method")


class TextToImage(DiffusionWorkloadConfig):
    """Text-to-image generation workload using open-image-preferences dataset."""
    
    def sample(self, num_requests: int) -> list[DiffusionRequest]:
        """Sample requests from the open-image-preferences dataset."""
        dataset = OpenPreferenceDataset(
            dataset_path="data-is-better-together/open-image-preferences-v1",
            dataset_split="cleaned",
            random_seed=self.seed,
        )
        requests = dataset.sample(num_requests=num_requests, batch_size=self.batch_size)
        return requests


class TextToVideo(DiffusionWorkloadConfig):
    """Text-to-video generation workload using EvalCrafter dataset."""
    
    # Video-specific parameters - will use model-specific defaults if not specified
    num_frames: Optional[int] = None
    fps: Optional[int] = None

    # For ConsisID, provide a URL or local path to reference face image
    img_file_path: Optional[str] = "https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/example_images/2.png?raw=true"
    
    @model_validator(mode="after")
    def _validate_video_workload(self) -> Self:
        """Apply video-specific model defaults."""
        # Call parent validator first
        super()._validate_workload()
        
        defaults = get_model_defaults(self.model_id)
        
        if self.num_frames is None:
            self.num_frames = defaults.get("num_frames", 16)
        if self.fps is None:
            self.fps = defaults.get("fps", 8)
                
        return self
    
    def sample(self, num_requests: int) -> list[DiffusionRequest]:
        """Sample requests from the EvalCrafter dataset."""
        dataset = EvalCrafterDataset(
            dataset_path="RaphaelLiu/EvalCrafter_T2V_Dataset",
            dataset_split="train",
            random_seed=self.seed,
        )
        requests = dataset.sample(num_requests=num_requests, batch_size=self.batch_size)
        return requests


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s[%(name)s:%(lineno)d] - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Example usage
    workload = TextToImage(
        base_dir=Path("run/diffusion/flux_t2i"),
        model_id="black-forest-labs/FLUX.1-dev",
        batch_size=10,
        height=512,
        width=512,
        inference_steps=28,
    )
    
    requests = workload.load_requests(warmup_iters=2, benchmark_iters=4)
    logger.info(f"Loaded {len(requests)} requests")
    for i, req in enumerate(requests[:3]):
        logger.info(f"Request {i+1}: {req.prompts[0][:100] if req.prompts else 'No prompts'}...") 