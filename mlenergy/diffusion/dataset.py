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
        batch_size: Number of images to generate in the batch.
        prompts: List of text prompts for batch generation.
    """
    batch_size: int
    prompts: list[str]


class OpenPreferenceDataset:
    """Open Image Preferences Dataset for diffusion model benchmarking."""

    def __init__(self, 
        dataset_path: str = "data-is-better-together/open-image-preferences-v1", 
        dataset_split: str = "cleaned", 
        random_seed: int = 42
    ) -> None:
        """Initialize the Open Preference dataset."""
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.random_seed = random_seed
        self.data = None

    def load_data(self):
        """Load data from HuggingFace datasets."""
        logger.info(f"Loading {self.dataset_path} dataset...")
        ds = load_dataset(self.dataset_path)
        return ds[self.dataset_split]

    def sample(self, num_requests: int, batch_size: int) -> list[DiffusionRequest]:
        """Sample requests from the open-image-preferences dataset.
        
        Args:
            num_requests: Number of DiffusionRequest objects to create.
            batch_size: Number of prompts per request.
            
        Returns:
            List of DiffusionRequest objects with sampled prompts.
        """
        if self.data is None:
            self.data = self.load_data()

        random.seed(self.random_seed)
        all_prompts = []
        for item in self.data:
            if isinstance(item, dict) and 'prompt' in item:
                prompt = item['prompt']
                if prompt is not None and prompt.strip():
                    all_prompts.append(str(prompt).strip())
        
        # Calculate total prompts needed
        total_prompts_needed = num_requests * batch_size
        
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
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            prompts_for_request = selected_prompts[start_idx:end_idx]
            
            request = DiffusionRequest(
                batch_size=batch_size,
                prompts=prompts_for_request,
            )
            requests.append(request)
        
        logger.info(f"Created {len(requests)} diffusion requests with {batch_size} prompts each")
        return requests
