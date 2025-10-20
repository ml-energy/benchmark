"""dLLM runtime configurations and implementations.

A runtime defines how a specific dLLM system (e.g., Fast-dLLM, MDLM, etc.)
should be initialized, configured, and used for generation.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import logging
import abc
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger("mlenergy.dllm.benchmark")


class DLLMRuntime(BaseModel, abc.ABC):
    """Base class for dLLM runtime configurations.

    A runtime configuration defines how a specific dLLM system should be
    initialized and used for generation.
    """

    @abc.abstractmethod
    def install_runtime(self) -> None:
        """Install the runtime dependencies and setup."""
        pass

    @abc.abstractmethod
    def run_batch(self, input_requests: list[str]) -> list[str]:
        """Run a batch of generation requests.

        Args:
            input_requests: List of input prompt strings.

        Returns:
            List of generated text outputs.
        """
        pass


class FastDLLMRuntime(DLLMRuntime):
    """Runtime implementation for Fast-dLLM (LLaDA).

    Attributes:
        model_id: Model identifier for the model to be used.
        steps: Number of sampling steps for generation.
        gen_length: Length of generated text.
        block_length: Block length for semi-autoregressive generation.
        temperature: Sampling temperature.
        mask_id: Token ID for the mask token.
        cache_mode: Caching mode ('none', 'prefix', or 'dual').
    """

    model_id: str = "GSAI-ML/LLaDA-8B-Instruct"
    steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    temperature: float = 0.0
    mask_id: int = 126336
    cache_mode: str = "dual"

    def install_runtime(self) -> None:
        """Install Fast-dLLM dependencies and setup Python path."""
        repo_url = "https://github.com/NVlabs/Fast-dLLM"
        commit = "8292f3c"
        base_dir = Path(__file__).resolve().parent
        fast_dir = base_dir / "Fast-dLLM"

        try:
            if not fast_dir.exists():
                logger.info("Cloning Fast-dLLM into %s", fast_dir)
                subprocess.run(["git", "clone", repo_url, str(fast_dir)], check=True)
                logger.info("Fetching and checking out commit %s in %s", commit, fast_dir)
                subprocess.run(["git", "fetch", "--all"], cwd=str(fast_dir), check=True)
                subprocess.run(["git", "checkout", commit], cwd=str(fast_dir), check=True)

            req_file = fast_dir / "requirements.txt"
            if req_file.exists():
                logger.info("Installing requirements from %s", req_file)
                subprocess.run(["uv", "pip", "install", "-r", str(req_file)], cwd=str(fast_dir), check=True)
            else:
                logger.warning("requirements.txt not found in %s; skipping pip install", fast_dir)

            # Add Fast-dLLM to Python path for imports
            if str(fast_dir) not in sys.path:
                sys.path.insert(0, str(fast_dir))
                logger.info("Added %s to Python path", fast_dir)

        except subprocess.CalledProcessError as e:
            logger.error("Failed to install Fast-dLLM: %s", e)
            raise

        logger.info("Fast-dLLM installation complete.")

    def run_batch(self, input_requests: list[str]) -> list[str]:
        """Run a batch of generation requests using Fast-dLLM.

        Args:
            input_requests: List of input prompt strings.

        Returns:
            List of generated text outputs.
        """
        


def default_fast_dllm_runtime() -> FastDLLMRuntime:
    """Create default Fast-dLLM runtime configuration."""
    return FastDLLMRuntime(
        model_id="GSAI-ML/LLaDA-8B-Instruct",
        device="cuda",
        steps=128,
        gen_length=128,
        block_length=32,
        cache_mode="dual",
    )