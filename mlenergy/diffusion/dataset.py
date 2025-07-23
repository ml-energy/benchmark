"""Request datasets for diffusion model benchmarks.

Similar to LLM datasets, this defines the structure for diffusion requests
and provides utilities for handling prompt datasets.
"""

from __future__ import annotations

import base64
import io
import logging
import random
from pathlib import Path
from typing import Any

from PIL import Image
from datasets import load_dataset
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DiffusionRequest(BaseModel):
    """Represents a batch diffusion inference request for benchmarking.

    Args:
        prompts: List of text prompts for batch generation.
        height: Output image height in pixels.
        width: Output image width in pixels.
        inference_steps: Number of denoising steps.
        seed: Random seed for generation.
        num_frames: Number of frames for video generation (optional).
        fps: Frames per second for video generation (optional).
    """
    batch_size: int
    prompts: list[str]
    height: int
    width: int
    inference_steps: int
    seed: int
    
    # Video-specific parameters (optional)
    num_frames: int | None = None
    fps: int | None = None


# def load_prompts_from_dataset(
#     batch_size: int,
#     split: str = "cleaned",
#     seed: int = 42,
#     dataset_name: str = "data-is-better-together/open-image-preferences-v1"
# ) -> list[str]:
#     """Load prompts from a text-to-image dataset.
    
#     Args:
#         batch_size: Number of prompts to sample
#         split: Dataset split to use
#         seed: Random seed for sampling
#         dataset_name: HuggingFace dataset name
        
#     Returns:
#         List of prompt strings
#     """
#     logger.info(f"Loading {batch_size} prompts from {dataset_name} ({split} split)")
    
#     # Load dataset
#     ds = load_dataset(dataset_name)
#     dataset = ds[split]
    
#     # Set random seed
#     random.seed(seed)
    
#     # Extract prompts
#     all_prompts = []
#     for item in dataset:
#         if isinstance(item, dict) and 'prompt' in item:
#             prompt = item['prompt']
#             if prompt is not None and prompt.strip():
#                 all_prompts.append(str(prompt).strip())
    
#     # Sample prompts
#     if batch_size > len(all_prompts):
#         logger.warning(f"Requested {batch_size} prompts but only {len(all_prompts)} available")
#         return all_prompts
    
#     selected_prompts = random.sample(all_prompts, batch_size)
#     logger.info(f"Selected {len(selected_prompts)} prompts")
    
#     return selected_prompts


# def create_diffusion_requests(
#     prompts: list[str],
#     height: int = 1024,
#     width: int = 1024,
#     inference_steps: int = 28,
#     guidance_scale: float = 3.5,
#     base_seed: int = 42,
#     num_frames: int | None = None,
#     fps: int | None = None
# ) -> list[DiffusionRequest]:
#     """Create DiffusionRequest objects from prompts.
    
#     Args:
#         prompts: List of text prompts
#         height: Output image height
#         width: Output image width  
#         inference_steps: Number of inference steps
#         guidance_scale: Guidance scale
#         base_seed: Base seed (each request gets base_seed + index)
#         num_frames: Number of frames for video generation
#         fps: Frames per second for video generation
        
#     Returns:
#         List of DiffusionRequest objects
#     """
#     requests = []
#     for i, prompt in enumerate(prompts):
#         request = DiffusionRequest(
#             prompt=prompt,
#             height=height,
#             width=width,
#             inference_steps=inference_steps,
#             guidance_scale=guidance_scale,
#             seed=base_seed + i,
#             num_frames=num_frames,
#             fps=fps
#         )
#         requests.append(request)
    
#     return requests


# if __name__ == "__main__":
#     # Example usage
#     prompts = load_prompts_from_dataset(batch_size=5, seed=42)
#     requests = create_diffusion_requests(prompts)
    
#     for i, req in enumerate(requests):
#         print(f"Request {i+1}: {req.prompt[:100]}...") 