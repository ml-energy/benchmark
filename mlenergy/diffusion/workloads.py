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
        request_path = self.to_path(of="requests")
        if request_path.exists():
            logger.info(f"Loading cached requests from {request_path}")
            with open(request_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Handle both single request list of requests
            if isinstance(data, list):
                requests = [DiffusionRequest(**req_data) for req_data in data]
            else:
                requests = [DiffusionRequest(**data)]
            return requests
        else:
            logger.info(f"Creating new requests for {self.model_id}")
            requests = self.sample(warmup_iters + benchmark_iters)
            if local_rank == 0:
                with open(request_path, 'w', encoding='utf-8') as f:
                    json.dump([req.model_dump() for req in requests], f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(requests)} requests to {request_path}")
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
        
        # Calculate total prompts needed
        total_prompts_needed = num_requests * self.batch_size
        
        # Sample prompts
        if total_prompts_needed > len(all_prompts):
            logger.warning(f"Requested {total_prompts_needed} total prompts but only {len(all_prompts)} prompts available")
            # Use all available prompts and repeat if needed
            selected_prompts = all_prompts * ((total_prompts_needed // len(all_prompts)) + 1)
            selected_prompts = selected_prompts[:total_prompts_needed]
        else:
            selected_prompts = random.sample(all_prompts, total_prompts_needed)
        
        # Create multiple DiffusionRequest objects
        requests = []
        for i in range(num_requests):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            prompts_for_request = selected_prompts[start_idx:end_idx]
            
            # Use different seed for each request to ensure variety
            request_seed = self.seed + i
            
            request = DiffusionRequest(
                batch_size=self.batch_size,
                prompts=prompts_for_request,
                height=self.height,
                width=self.width,
                inference_steps=self.inference_steps,
                seed=request_seed,
            )
            requests.append(request)
        
        logger.info(f"Created {len(requests)} T2I diffusion requests with {self.batch_size} prompts each")
        return requests


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
    
    requests = workload.load_requests(warmup_iters=2, benchmark_iters=4)
    logger.info(f"Loaded {len(requests)} requests")
    for i, req in enumerate(requests[:3]):
        logger.info(f"Request {i+1}: {req.prompts[0][:100] if req.prompts else 'No prompts'}...") 