"""Configuration loading for model and GPU-specific vLLM settings.

This module provides functionality to load model-specific and GPU-specific
configurations for vLLM server deployment. Configurations are stored in YAML
files organized by model ID and GPU model.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)


def get_vllm_config_path(
    model_id: str,
    gpu_model: str,
    workload: str,
    mode: Literal["monolithic", "prefill", "decode"],
    config_base_dir: Path | str = "configs/vllm",
) -> Path:
    """Get the path to vLLM config file for a given model, GPU, and deployment mode.

    Args:
        model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        gpu_model: GPU model name (e.g., "H100", "A100", "B200").
        workload: Name of the workload (e.g., "lm-arena-chat").
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
    model_config_dir = config_base_path / workload / model_id / gpu_model
    config_file = model_config_dir / f"{mode}.config.yaml"
    config_file = config_file.absolute()

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
    workload: str,
    mode: Literal["monolithic", "prefill", "decode"],
    config_base_dir: Path | str = "configs/vllm",
) -> dict[str, str]:
    """Load environment variables for a given model, GPU, and deployment mode.

    Args:
        model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        gpu_model: GPU model name (e.g., "H100", "A100", "B200").
        workload: Name of the workload (e.g., "lm-arena-chat").
        mode: Deployment mode ("monolithic", "prefill", or "decode").
        config_base_dir: Base directory for configuration files.

    Returns:
        Dictionary of environment variables. Returns empty dict if file doesn't exist.
    """
    config_base_path = Path(config_base_dir)

    # Construct path to env config file
    model_config_dir = config_base_path / workload / model_id / gpu_model
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


def load_extra_body(
    model_id: str,
    gpu_model: str,
    workload: str,
    config_base_dir: Path | str = "configs/vllm",
) -> dict[str, str]:
    """Load extra body content for a given model, GPU, and deployment mode.

    Args:
        model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        gpu_model: GPU model name (e.g., "H100", "A100", "B200").
        workload: Name of the workload (e.g., "lm-arena-chat").
        config_base_dir: Base directory for configuration files.

    Returns:
        Extra body kwargs as a dictionary. Empty dict if the file doesn't exist.
    """
    config_base_path = Path(config_base_dir)

    # Construct path to extra body file
    model_config_dir = config_base_path / workload / model_id / gpu_model
    extra_body_file = model_config_dir / "extra_body.json"

    if not extra_body_file.exists():
        logger.info(
            "No extra body file found at %s (optional)",
            extra_body_file,
        )
        return {}

    with open(extra_body_file) as f:
        extra_body = json.load(f)

    logger.info("Extra request body kwargs: %s", extra_body)

    return extra_body
