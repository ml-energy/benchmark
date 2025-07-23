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
