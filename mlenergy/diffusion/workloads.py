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
from typing import Any, Literal, Self

from datasets import load_dataset
from pydantic import BaseModel, model_validator

from mlenergy.diffusion.dataset import DiffusionRequest
from mlenergy.constants import DEFAULT_SEED

logger = logging.getLogger(__name__)


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
    """

    # Input parameters
    base_dir: Path
    seed: int = DEFAULT_SEED
    model_id: str
    batch_size: int
    
    # Generation parameters
    height: int = 1024
    width: int = 1024
    inference_steps: int = 28

    # Optimization parameters
    use_torch_compile: bool = False

    @model_validator(mode="after")
    def _validate_workload(self) -> Self:
        """Validate the sanity of the workload."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
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
            f"tc-{self.use_torch_compile}",
        ]
        return parts

    def load_requests(self) -> DiffusionRequest:
        """Load the requests from the file specified by the configuration.

        If the file does not exist, it will call `sample` to sample new requests.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        request_path = self.to_path(of="requests")
        if request_path.exists():
            logger.info(f"Loading cached requests from {request_path}")
            with open(request_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            request = DiffusionRequest(**data)
            return request
        else:
            logger.info(f"Creating new requests for {self.model_id}")
            request = self.sample()
            if local_rank == 0:
                with open(request_path, 'w', encoding='utf-8') as f:
                    json.dump(request.model_dump(), f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {request.batch_size} requests to {request_path}")
            return request

    def sample(self) -> DiffusionRequest:
        """Sample requests from the dataset.
        
        This method should be implemented by subclasses to specify dataset-specific sampling logic.
        Returns a single DiffusionRequest containing a batch of prompts.
        """
        raise NotImplementedError("Subclasses must implement the sample method")


class TextToImage(DiffusionWorkloadConfig):
    """Text-to-image generation workload using open-image-preferences dataset."""
    
    def sample(self) -> DiffusionRequest:
        """Sample requests from the open-image-preferences dataset."""
        logger.info("Loading open-image-preferences-v1 dataset...")
        ds = load_dataset("data-is-better-together/open-image-preferences-v1")
        random.seed(self.seed)
        dataset = ds["cleaned"]
        all_prompts = []
        for item in dataset:
            if isinstance(item, dict) and 'prompt' in item:
                prompt = item['prompt']
                if prompt is not None and prompt.strip():
                    all_prompts.append(str(prompt).strip())
        
        # Sample prompts
        if self.batch_size > len(all_prompts):
            logger.warning(f"Requested {self.batch_size} requests but only {len(all_prompts)} prompts available")
            batch_size = len(all_prompts)
        else:
            batch_size = self.batch_size
            
        selected_prompts = random.sample(all_prompts, batch_size)
        
        request = DiffusionRequest(
            batch_size=self.batch_size,
            prompts=selected_prompts,
            height=self.height,
            width=self.width,
            inference_steps=self.inference_steps,
            seed=self.seed,
        )
        logger.info(f"Created T2I diffusion request with {len(selected_prompts)} prompts")
        return request


class TextToVideo(DiffusionWorkloadConfig):
    """Text-to-video generation workload using EvalCrafter dataset."""
    
    # Video-specific parameters
    num_frames: int = 16
    fps: int = 8


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
    
    requests = workload.load_requests()
    logger.info(f"Loaded {len(requests)} requests")
    for i, req in enumerate(requests[:3]):
        logger.info(f"Request {i+1}: {req.prompt[:100]}...") 