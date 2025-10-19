"""Configuration loading for model and GPU-specific vLLM settings.

This module provides functionality to load model-specific and GPU-specific
configurations for vLLM server deployment. Configurations are stored in YAML
files organized by model ID and GPU model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)


def get_vllm_config_path(
    model_id: str,
    gpu_model: str,
    mode: Literal["monolithic", "prefill", "decode"],
    config_base_dir: Path | str = "configs/vllm",
) -> Path:
    """Get the path to vLLM config file for a given model, GPU, and deployment mode.

    Args:
        model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        gpu_model: GPU model name (e.g., "H100", "A100", "B200").
        mode: Deployment mode ("monolithic", "prefill", or "decode").
        config_base_dir: Base directory for configuration files.

    Returns:
        Path to the vLLM config YAML file.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
    """
    config_base_path = Path(config_base_dir)

    # Construct path to config file
    # model_id might contain slashes (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    model_config_dir = config_base_path / model_id / gpu_model
    config_file = model_config_dir / f"{mode}.config.yaml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"vLLM configuration file not found: {config_file}\n"
            f"Expected config for model={model_id}, gpu={gpu_model}, mode={mode}"
        )

    logger.info("Found vLLM config at %s", config_file)
    return config_file


def load_env_vars(
    model_id: str,
    gpu_model: str,
    mode: Literal["monolithic", "prefill", "decode"],
    config_base_dir: Path | str = "configs/vllm",
) -> dict[str, str]:
    """Load environment variables for a given model, GPU, and deployment mode.

    Args:
        model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        gpu_model: GPU model name (e.g., "H100", "A100", "B200").
        mode: Deployment mode ("monolithic", "prefill", or "decode").
        config_base_dir: Base directory for configuration files.

    Returns:
        Dictionary of environment variables. Returns empty dict if file doesn't exist.
    """
    config_base_path = Path(config_base_dir)

    # Construct path to env config file
    model_config_dir = config_base_path / model_id / gpu_model
    env_config_file = model_config_dir / f"{mode}.env.yaml"

    if not env_config_file.exists():
        logger.info(
            "No environment config found at %s (optional)",
            env_config_file,
        )
        return {}

    logger.info("Loading environment config from %s", env_config_file)
    with open(env_config_file) as f:
        env_vars = yaml.safe_load(f) or {}

    # Ensure all values are strings
    return {k: str(v) for k, v in env_vars.items()}
