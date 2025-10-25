"""dLLM runtime configurations and implementations.

A runtime defines how a specific dLLM system (e.g., Fast-dLLM, MDLM, etc.)
should be initialized, configured, and used for generation.
"""

from __future__ import annotations

import abc
import logging

import torch
import types
from pydantic import BaseModel
from transformers import AutoTokenizer

from fast_dllm.dream.model.modeling_dream import DreamModel
from fast_dllm.dream.model.generation_utils_block import DreamGenerationMixin
from fast_dllm.llada.model.modeling_llada import LLaDAModelLM
from fast_dllm.llada.generate import (
    generate,
    generate_with_prefix_cache,
    generate_with_dual_cache,
)


logger = logging.getLogger("mlenergy.dllm.benchmark")


class DLLMRuntime(BaseModel, abc.ABC):
    """Base class for dLLM runtime configurations.

    A runtime configuration defines how a specific dLLM system should be
    initialized and used for generation.

    Attributes:
        model_id: Model identifier for the model to be used.
        steps: Number of sampling/diffusion steps for generation.
        gen_length: Length of generated text.
        block_length: Block length for semi-autoregressive generation.
        mask_id: Token ID for the mask token.
        cache_mode: Caching mode ('none', 'prefix', or 'dual').
        remasking: Remasking strategy ('low_confidence' or 'random').
    """

    model_id: str
    steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    cache_mode: str | None = None
    remasking: str = "low_confidence"

    model: object | None = None
    tokenizer: object | None = None

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer for the runtime."""
        pass

    @abc.abstractmethod
    def run_one_batch(self, prompts: list[str]) -> list[str]:
        """Run a batch of generation requests.

        Args:
            prompts: List of input prompt strings.

        Returns:
            List of generated text outputs.
        """
        pass


class LladaRuntime(DLLMRuntime):
    """Runtime implementation for Fast-dLLM (LLaDA).

    All configuration parameters are inherited from DLLMRuntime.
    """

    model_id: str = "GSAI-ML/LLaDA-8B-Instruct"

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.load_model()

    def load_model(self) -> None:
        """
        Loading model and tokenizer
        """
        device = "cuda"
        self.model = (
            LLaDAModelLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def run_one_batch(self, prompts: list[str]) -> list[str]:
        device = "cuda"

        formatted_prompts = []
        for prompt in prompts:
            m = [{"role": "user", "content": prompt}]
            user_input = self.tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            formatted_prompts.append(user_input)

        tokenized = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)

        logger.info(
            f"Batch size: {input_ids.shape[0]}, Max length: {input_ids.shape[1]}"
        )

        answers = []
        if self.cache_mode is None:
            out, nfe = generate(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0.0,
                remasking=self.remasking,
                threshold=None,
            )
            answers = self.tokenizer.batch_decode(
                out[:, input_ids.shape[1] :], skip_special_tokens=True
            )
        elif self.cache_mode == "prefix":
            out, nfe = generate_with_prefix_cache(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0.0,
                remasking=self.remasking,
            )
            answers = self.tokenizer.batch_decode(
                out[:, input_ids.shape[1] :], skip_special_tokens=True
            )
        elif self.cache_mode == "dual":
            out, nfe = generate_with_dual_cache(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0.0,
                remasking=self.remasking,
            )
            answers = self.tokenizer.batch_decode(
                out[:, input_ids.shape[1] :], skip_special_tokens=True
            )

        logger.info(f"Generated {len(answers)} outputs with {nfe} function evaluations")

        print(answers)
        return answers


class DreamRuntime(DLLMRuntime):
    """Runtime implementation for Fast-dLLM (DREAM).

    All configuration parameters are inherited from DLLMRuntime.
    Default values are set for DREAM-specific behavior.
    """

    model_id: str = "Dream-org/Dream-v0-Instruct-7B"
    steps: int = 16

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.load_model()

    def load_model(self) -> None:
        """
        Loading DREAM model and tokenizer with block-based generation.
        """

        device = "cuda"
        logger.info("Loading DREAM model: %s", self.model_id)

        self.model = (
            DreamModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        # Patch model with block-based generation methods
        # Modeled after fast-dllm's examples
        self.model.diffusion_generate = types.MethodType(
            DreamGenerationMixin.diffusion_generate, self.model
        )
        self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)

        logger.info("DREAM model loaded successfully")

    def run_one_batch(self, prompts: list[str]) -> list[str]:
        """Run generation for prompts using DREAM.

        Note: DREAM does not support batching. This method only accepts a single prompt.

        Args:
            prompts: List of input prompt strings (must have length 1).

        Returns:
            List of generated text outputs (length 1).

        Raises:
            ValueError: If prompts list has length > 1.
        """
        if len(prompts) > 1:
            raise ValueError(
                f"DreamRuntime does not support batching. "
                f"Received {len(prompts)} prompts, but only batch_size=1 is supported. "
                f"Please set --workload.batch-size=1 when using DreamRuntime."
            )

        device = "cuda"
        prompt = prompts[0]

        m = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            m, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        logger.info(f"Input length: {input_ids.shape[1]}")

        output = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.gen_length,
            time_steps=self.steps,
            block_size=self.block_length,
        )

        answer = self.tokenizer.batch_decode(
            output[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        logger.info("Generated 1 output")

        return [answer]


def default_llada_runtime() -> LladaRuntime:
    """Create default LLaDA runtime configuration."""
    return LladaRuntime(
        model_id="GSAI-ML/LLaDA-8B-Instruct",
        steps=128,
        gen_length=128,
        block_length=32,
        cache_mode="dual",
        remasking="low_confidence",
    )


def default_dream_runtime() -> DreamRuntime:
    """Create default DREAM runtime configuration."""
    return DreamRuntime(
        model_id="Dream-org/Dream-v0-Instruct-7B",
        steps=16,
        gen_length=128,
        block_length=32,
        cache_mode="dual",
        remasking="low_confidence",
    )
