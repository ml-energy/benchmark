"""dLLM runtime configurations and implementations.

A runtime defines how a specific dLLM system (e.g., Fast-dLLM, MDLM, etc.)
should be initialized, configured, and used for generation.
"""

from __future__ import annotations

import abc
import logging
from typing import Literal

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


logger = logging.getLogger(__name__)


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
    gen_length: int = 256
    block_length: int = 32
    cache_mode: str | None = None
    remasking: Literal["low_confidence", "random"] = "low_confidence"

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


class FastDLLMRuntime(DLLMRuntime):

    """Runtime implementation for Fast-dLLM."""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.load_model()

    def load_model(self) -> None:
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
        if self.model_id == "GSAI-ML/LLaDA-8B-Instruct":
            self._run_one_batch_llada(prompts)
        elif self.model_id == "Dream-org/Dream-v0-Instruct-7B":
            self._run_one_batch_dream(prompts)
        else:
            raise ValueError(f"Unsupported model_id: {self.model_id}")

   

    def _run_one_batch_llada(self, prompts: list[str]) -> list[str]:
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
            out, _ = generate(
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
            out, _ = generate_with_prefix_cache(
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
            out, _ = generate_with_dual_cache(
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

        return answers

    def _run_one_batch_dream(self, prompts: list[str]) -> list[str]:
        """Note: DREAM does not support batching. This method only accepts a single prompt.

        Raises:
            ValueError: If prompts list has length > 1.
        """
        if len(prompts) > 1:
            raise ValueError(
                f"DreamRuntime does not support batching. "
                f"Received {len(prompts)} prompts, but only batch_size=1 is supported. "
                f"Please set --workload.batch-size=1 when using DreamRuntime."
            )

        self.model.diffusion_generate = types.MethodType(
            DreamGenerationMixin.diffusion_generate, self.model
        )

        self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)
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


def default_dllm_runtime() -> DLLMRuntime:
    """Create default LLaDA runtime configuration."""
    return FastDLLMRuntime(
        model_id="GSAI-ML/LLaDA-8B-Instruct",
        steps=128,
        gen_length=128,
        block_length=32,
        cache_mode="dual",
        remasking="low_confidence",
    )
