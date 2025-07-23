"""Diffusion model benchmarking package.

This package provides comprehensive benchmarking tools for diffusion models,
including energy monitoring and performance metrics collection.
"""

from .dataset import DiffusionRequest
from .workloads import DiffusionWorkloadConfig, TextToImage, TextToVideo
from .benchmark import DiffusionArgs

__all__ = [
    "DiffusionRequest",
    "DiffusionWorkloadConfig",
    "TextToImage",
    "TextToVideo",
    "DiffusionArgs",
] 