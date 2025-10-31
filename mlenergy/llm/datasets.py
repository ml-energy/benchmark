"""Request datasets for LLM and MLLM benchmarks.

A dataset encloses logic for downloading and preparing request datasets,
sampling a specific number of requests, and saving it to a file.

Inspired by https://github.com/vllm-project/vllm/blob/7ba34b12/vllm/benchmarks/datasets.py
"""

from __future__ import annotations

import base64
import io
import logging
import random
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from scipy import stats
from PIL import Image
from datasets import (
    get_dataset_config_names,
    get_dataset_split_names,
    interleave_datasets,
    load_dataset,
)
from pydantic import BaseModel

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class SampleRequest(BaseModel):
    """Represents a single inference request for benchmarking.

    Args:
        prompt: When it's a `str`, it's the input text prompt for the model.
            When it's a `list[str]`, it's the history of a multi-turn conversation.
        prompt_token_ids: Optional list of token IDs for the prompt. Used for models/tasks that need
            pre-tokenized inputs.
        completion: The expected output text from the model.
        prompt_len: The length of the prompt in tokens.
        expected_output_len: The expected length of the output in tokens.
        multimodal_contents: A list of dictionaries containing multimodal content for OpenAI Chat Completion.
        multimodal_content_paths: A list of paths to the original multimodal data files (empty if not dumped).
    """

    prompt: str | list[str]
    prompt_token_ids: list[int] | None = None
    completion: str
    prompt_len: int
    expected_output_len: int
    multimodal_contents: list[dict[str, Any]]
    multimodal_content_paths: list[str] = []


def maybe_oversample_requests(
    requests: list[SampleRequest], num_requests: int, random_seed: int
) -> None:
    """Oversamples the list of requests if its size is less than the desired number.

    Args:
        requests: The current list of sampled requests.
        num_requests: The target number of requests.
        random_seed: Random seed for reproducible oversampling.
    """
    if len(requests) < num_requests:
        logger.warning("Oversampling requests to reach %d total samples.", num_requests)
        random.seed(random_seed)
        additional = random.choices(requests, k=num_requests - len(requests))
        requests.extend(additional)
        logger.info("Oversampled requests to reach %d total samples.", num_requests)


def process_audio_bytes(data: bytes | io.BytesIO) -> dict[str, Any]:
    """Convert raw audio bytes to a multimodal content dictionary.

    Args:
        data: Raw audio data as bytes or a byte stream.

    Returns:
        A dictionary containing the base64-encoded audio URL.
    """
    if isinstance(data, io.BytesIO):
        data = data.getvalue()
    audio_base64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "audio_url",
        "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
    }


def process_video_bytes(data: bytes) -> dict[str, Any]:
    """Convert raw video bytes to a multimodal content dictionary.

    Args:
        data: Raw video data as bytes.

    Returns:
        A dictionary containing the base64-encoded video URL.
    """
    video_base64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
    }


def process_image(image: dict | Image.Image | str) -> dict[str, Any]:
    """Process a single image input and return a multimedia content dictionary.

    Supports three input types:

    1. Dictionary with raw image bytes:
        - Expects a dict with a 'bytes' key containing raw image data.
        - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input:
        - Converts the image to RGB.
        - Saves the image as a JPEG in memory.
        - Encodes the JPEG data as a base64 string.
        - Returns a dictionary with the image as a base64 data URL.

    3. String input:
        - Treats the string as a URL or local file path.
        - Prepends "file://" if the string doesn't start with "http://" or "file://".
        - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(image, dict):
        image = Image.open(io.BytesIO(image["bytes"]))

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        }

    if isinstance(image, str):
        image_url = (
            image if image.startswith(("http://", "file://")) else f"file://{image}"
        )
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(
        f"Invalid image input {image} with type {type(image)}. Must be a PIL.Image.Image"
        " or str or dictionary with raw image bytes."
    )


class VisionArenaDataset:
    """Vision Arena Dataset."""

    def __init__(self, dataset_path: str, dataset_split: str, random_seed: int) -> None:
        """Initialize the Vision Arena dataset."""
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.random_seed = random_seed
        self.data: Any = None

    def load_data(self):
        """Load data from HuggingFace datasets.

        This dataset is entirely hosted on Hugging Face Hub.
        """
        data = load_dataset(
            self.dataset_path,
            split=self.dataset_split,
            streaming=True,
        )
        return data.shuffle(seed=self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        num_images: int = 1,
        dump_multimodal_dir: Path | None = None,
    ) -> list[SampleRequest]:
        """Sample requests from the dataset."""
        if self.data is None:
            self.data = self.load_data()

        # Use examples with at least 2 conversation entries (prompt and model response)
        filtered_data = self.data.filter(lambda x: len(x["conversation"]) >= 2)

        sampled_requests = []
        for item in filtered_data:
            assert isinstance(item, dict), (
                "Each item in the dataset must be a dictionary."
            )
            if len(sampled_requests) >= num_requests:
                break

            prompt = item["conversation"][0][0]["content"]
            model_response = item["conversation"][1][0]["content"]

            # Process image content
            mm_content = process_image(item["images"][0])

            # Handle image dumping if requested
            image_paths = []
            if dump_multimodal_dir is not None:
                image_id = item["images"][0]["path"]
                image_path: Path = dump_multimodal_dir / image_id
                if not image_path.exists():
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(item["images"][0]["bytes"])
                image_paths = [str(image_path)] * num_images

            prompt_len = len(tokenizer(prompt).input_ids)
            expected_output_len = len(tokenizer(model_response).input_ids)

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    completion=model_response,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len,
                    multimodal_contents=[mm_content] * num_images,
                    multimodal_content_paths=image_paths,
                )
            )

        maybe_oversample_requests(sampled_requests, num_requests, self.random_seed)

        return sampled_requests


class LLaVAVideoDataset:
    """LLaVA Video Dataset."""

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        random_seed: int,
        video_data_dir: str,
    ) -> None:
        """Initialize the LLaVA Video dataset."""
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.random_seed = random_seed
        self.video_data_dir = Path(video_data_dir)
        self.data = None

    def load_data(self):
        """Load data from HuggingFace datasets."""
        streams = []

        for subset in get_dataset_config_names(self.dataset_path):
            if self.dataset_split in get_dataset_split_names(
                self.dataset_path, config_name=subset
            ):
                streams.append(
                    load_dataset(
                        self.dataset_path,
                        name=subset,
                        split=self.dataset_split,
                        streaming=True,
                    )
                )
            else:
                logger.info("Skipping %s (no '%s' split)", subset, self.dataset_split)

        data = interleave_datasets(streams, seed=self.random_seed)
        return data.shuffle(seed=self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        num_videos: int = 1,
        dump_multimodal_dir: Path | None = None,
    ) -> list[SampleRequest]:
        """Sample requests from the dataset."""
        if self.data is None:
            self.data = self.load_data()

        # Use examples with at least 2 conversation entries (prompt and model response)
        filtered_data = self.data.filter(lambda x: len(x["conversations"]) >= 2)

        sampled_requests = []

        for item in filtered_data:
            assert isinstance(item, dict), (
                "Each item in the dataset must be a dictionary."
            )
            if len(sampled_requests) >= num_requests:
                break

            if "video" not in item or not item["video"]:
                logger.warning("Skipping item %s with missing video path.", item["id"])
                continue

            mm_data_id = item["video"]

            extracted_path = self.video_data_dir / mm_data_id
            if not extracted_path.exists():
                logger.warning(
                    "Video file path %s does not exist. Skipping item.", extracted_path
                )
                continue

            mm_content = process_video_bytes(extracted_path.read_bytes())

            video_paths = []
            if dump_multimodal_dir is not None:
                video_path: Path = dump_multimodal_dir / mm_data_id
                if not video_path.exists():
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    video_path.write_bytes(extracted_path.read_bytes())
                video_paths = [str(video_path)] * num_videos

            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]
            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    completion=completion,
                    prompt_len=prompt_len,
                    expected_output_len=completion_len,
                    multimodal_contents=[mm_content] * num_videos,
                    multimodal_content_paths=video_paths,
                )
            )

        maybe_oversample_requests(sampled_requests, num_requests, self.random_seed)

        return sampled_requests


class AudioSkillsDataset:
    """Audio Skills Dataset."""

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        random_seed: int,
        audio_data_dir: str,
    ) -> None:
        """Initialize the Audio Skills dataset."""
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.random_seed = random_seed
        self.audio_data_dir = Path(audio_data_dir)
        self.data = None

    def load_data(self):
        """Load data from HuggingFace datasets."""
        data = load_dataset(self.dataset_path, split=self.dataset_split, streaming=True)
        return data.shuffle(seed=self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        num_audio: int = 1,
        dump_multimodal_dir: Path | None = None,
    ) -> list[SampleRequest]:
        """Sample requests from the dataset."""
        if self.data is None:
            self.data = self.load_data()

        # Use examples with at least 2 conversation entries (prompt and model response)
        filtered_data = self.data.filter(lambda x: len(x["conversations"]) >= 2)

        sampled_requests = []
        for item in filtered_data:
            assert isinstance(item, dict), (
                "Each item in the dataset must be a dictionary."
            )
            if len(sampled_requests) >= num_requests:
                break

            mm_data_id = item["sound"]
            extracted_path = self.audio_data_dir / mm_data_id

            # mm_content_bytes = self._get_audio_bytes(item["sound"])
            if not extracted_path.exists():
                logger.warning(
                    "Audio file path %s does not exist. Skipping item.", extracted_path
                )
                continue

            mm_content = process_audio_bytes(extracted_path.read_bytes())

            audio_paths = []
            if dump_multimodal_dir is not None:
                audio_path: Path = dump_multimodal_dir / mm_data_id
                if not audio_path.exists():
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    audio_path.write_bytes(extracted_path.read_bytes())
                audio_paths = [str(audio_path)] * num_audio

            conversations = item["conversations"]
            prompt, completion = conversations[0]["value"], conversations[1]["value"]
            prompt_len = len(tokenizer(prompt).input_ids)
            expected_output_len = len(tokenizer(completion).input_ids)

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    completion=completion,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len,
                    multimodal_contents=[mm_content] * num_audio,
                    multimodal_content_paths=audio_paths,
                )
            )

        maybe_oversample_requests(sampled_requests, num_requests, self.random_seed)

        return sampled_requests


class OmniDataset:
    """Dataset with Video, Audio, Image, and Text."""

    def __init__(
        self,
        video_dataset_path: str,
        video_dataset_split: str,
        video_data_dir: str,
        random_seed: int = 0,
    ) -> None:
        self.random_seed = random_seed

        self.video_dataset = LLaVAVideoDataset(
            dataset_path=video_dataset_path,
            dataset_split=video_dataset_split,
            random_seed=self.random_seed,
            video_data_dir=video_data_dir,
        )

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        num_images: int = 1,
        num_videos: int = 1,
        num_audio: int = 1,
        dump_multimodal_dir: Path | None = None,
    ) -> list[SampleRequest]:
        raise NotImplementedError()


class LMArenaHumanPreferenceDataset:
    """LMArena Human Preference dataset for text-only chat."""

    def __init__(
        self, dataset_path: str, dataset_split: str, random_seed: int = 0
    ) -> None:
        """Initialize the LMArena Human Preference dataset."""
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.random_seed = random_seed
        self.data = None

    def load_data(self):
        """Load data from HuggingFace datasets."""
        data = load_dataset(self.dataset_path, split=self.dataset_split)
        return data.shuffle(seed=self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
    ) -> list[SampleRequest]:
        if self.data is None:
            self.data = self.load_data()

        requests: list[SampleRequest] = []

        # One request can have more than one turns. So, one request will
        # create `turn` many requests, where each turn is the concatenation
        # of all historical user prompts and model responses.
        for item in self.data:
            assert isinstance(item, dict), (
                "Each item in the dataset must be a dictionary."
            )

            num_turns = item["turn"]
            conversation = item["conversation_a"]

            for turns in range(num_turns):
                if len(requests) >= num_requests:
                    break

                messages = []
                prompt_len = 0
                for message in conversation[: 2 * turns + 1]:
                    content = message["content"]
                    messages.append(content)
                    prompt_len += len(tokenizer(content).input_ids)
                completion = conversation[2 * turns + 1]["content"]
                completion_len = len(tokenizer(completion).input_ids)

                requests.append(
                    SampleRequest(
                        prompt=messages,
                        completion=completion,
                        prompt_len=prompt_len,
                        expected_output_len=completion_len,
                        multimodal_contents=[],
                    )
                )

        maybe_oversample_requests(requests, num_requests, self.random_seed)

        return requests


def render_fim_prompt(
    prefix: str, suffix: str, tokenizer: PreTrainedTokenizerBase
) -> tuple[str, list[int] | None]:
    """Render the fill-in-the-middle prompt.

    Returns:
        A tuple of (prompt string, optional list of token IDs).
    """
    model = tokenizer.name_or_path.lower()

    if (
        model.startswith("qwen/qwen2.5-coder")
        or model.startswith("qwen/qwen3-coder")
        or model.startswith("google/codegemma")
    ):
        return (
            f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
            None,
        )

    if model.startswith("deepseek-ai/deepseek-coder-v2"):
        return (
            f"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>",
            None,
        )

    if model.startswith("mistralai/codestral"):
        # Special case for Codestral; it'll be mlenergy.llm.workloads.CodestralTokenizer.
        tokenized = tokenizer(prefix=prefix, suffix=suffix)
        return (tokenized.text, tokenized.input_ids)

    raise NotImplementedError(
        f"Unsupported model {model} for fill-in-the-middle prompt rendering."
    )


class SourcegraphFIMDataset:
    """Sourcegraph fill-in-the-middle dataset."""

    def __init__(self, dataset_path: str, dataset_split: str, random_seed: int) -> None:
        """Initialize the dataset."""
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.random_seed = random_seed
        self.data: Any = None

    def load_data(self):
        """Load data from HuggingFace Hub."""
        data = load_dataset(self.dataset_path, split=self.dataset_split)
        return data.shuffle(seed=self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
    ) -> list[SampleRequest]:
        if self.data is None:
            self.data = self.load_data()

        random.seed(self.random_seed)

        requests: list[SampleRequest] = []
        for item in self.data:
            if len(requests) >= num_requests:
                break

            assert isinstance(item, dict), (
                "Each item in the dataset must be a dictionary."
            )

            prefix, answer, suffix = item["prefix"], item["middle"], item["suffix"]
            prompt, input_ids = render_fim_prompt(prefix, suffix, tokenizer)

            answer_len = len(tokenizer(answer).input_ids)

            # Some tokenizers return a list of token IDs directly (e.g., CodestralTokenizer).
            if input_ids is None:
                prompt_len = len(tokenizer(prompt).input_ids)
            else:
                prompt_len = len(input_ids)

            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_token_ids=input_ids,
                    completion=answer,
                    prompt_len=prompt_len,
                    expected_output_len=answer_len,
                    multimodal_contents=[],
                )
            )

        maybe_oversample_requests(requests, num_requests, self.random_seed)

        return requests


class GPQADataset:
    """GPQA dataset."""

    def __init__(
        self,
        dataset_path: str,
        dataset_subset: str,
        dataset_split: str,
        random_seed: int,
    ) -> None:
        """Initialize the dataset."""
        self.dataset_path = dataset_path
        self.dataset_subset = dataset_subset
        self.dataset_split = dataset_split
        self.random_seed = random_seed
        self.data: Any = None

    def load_data(self):
        data = load_dataset(
            self.dataset_path,
            self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        return data.shuffle(seed=self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
    ) -> list[SampleRequest]:
        if self.data is None:
            self.data = self.load_data()

        random.seed(self.random_seed)

        requests: list[SampleRequest] = []
        for item in self.data:
            if len(requests) >= num_requests:
                break

            assert isinstance(item, dict), (
                "Each item in the dataset must be a dictionary."
            )

            # Shuffle choices
            choices = [
                item["Incorrect Answer 1"].strip(),
                item["Incorrect Answer 2"].strip(),
                item["Incorrect Answer 3"].strip(),
                item["Correct Answer"].strip(),
            ]
            answer = choices[-1]
            random.shuffle(choices)

            question = item["Question"]
            prompt = f"What is the correct answer to the following question: {question}\n\nChoices:"
            for letter, choice in zip("ABCD", choices, strict=True):
                prompt += f"\n({letter}) {choice}"

            answer_letter = choices.index(answer)
            completion = f"({answer_letter}) " + answer

            prompt_len = len(tokenizer(prompt).input_ids)
            comp_len = len(tokenizer(completion).input_ids)

            requests.append(
                SampleRequest(
                    prompt=prompt,
                    completion=completion,
                    prompt_len=prompt_len,
                    expected_output_len=comp_len,
                    multimodal_contents=[],
                )
            )

        maybe_oversample_requests(requests, num_requests, self.random_seed)

        return requests


class ParetoExpDistributionDataset:
    """Dataset that generates random strings with controlled input/output token lengths.

    This dataset samples input lengths from a Pareto distribution and output lengths
    from an Exponential distribution, and generates synthetic random text.
    """

    def __init__(
        self,
        random_seed: int = 0,
        input_mean: float = 500.0,
        output_mean: float = 300.0,
        pareto_a: float = 2.5,
        model_max_length: int = 32768,
    ) -> None:
        """Initialize the ParetoExpDistributionDataset dataset.

        Args:
            random_seed: Random seed for reproducible sampling.
            input_mean: Mean number of input tokens for Pareto distribution.
            output_mean: Mean number of output tokens for Exponential distribution.
            pareto_a: Shape parameter for Pareto distribution.
                Smaller pareto_a (closer to 1): Heavier tail → more extreme/large values
                Larger pareto_a: Lighter tail → values concentrated around smaller numbers
        """
        self.random_seed = random_seed
        self.input_mean = input_mean
        self.pareto_a = pareto_a
        self.output_mean = output_mean
        # some are too long, so we limit it to 32768
        self.max_length = min(model_max_length, 32768)

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Generate Pareto distribution for input tokens
        # For Pareto, mean = a * b / (a-1) where a > 1
        # We use a = 2.5, then b = mean * (a-1)/a = mean * 1.5/2.5 = mean * 0.6
        pareto_b = self.input_mean * (self.pareto_a - 1) / self.pareto_a
        input_pdf = stats.pareto.pdf(
            np.arange(self.max_length), self.pareto_a, scale=pareto_b
        )
        self.input_pdf = input_pdf / np.sum(input_pdf)  # Normalize to sum to 1

        # Generate Exponential distribution for output tokens
        # For Exponential, mean = 1/lambda
        exp_lambda = 1 / self.output_mean
        output_pdf = stats.expon.pdf(np.arange(self.max_length), scale=1 / exp_lambda)
        self.output_pdf = output_pdf / np.sum(output_pdf)  # Normalize to sum to 1

        self.rng = np.random.default_rng(random_seed)

    def _generate_random_text_with_length(
        self, tokenizer: PreTrainedTokenizerBase, target_length: int
    ) -> str:
        """Generate random text that tokenizes to approximately the target length.

        Args:
            tokenizer: Tokenizer to use for measuring token length.
            target_length: Target number of tokens.

        Returns:
            Random text string that tokenizes to approximately target_length tokens.
        """
        special_ids = set(tokenizer.all_special_ids)
        vocab_size = len(tokenizer)

        # Pre-generate a long sequence of random token IDs
        pool_size = max(self.max_length, target_length * 2)
        long_token_ids = []

        while len(long_token_ids) < pool_size:
            # Generate extra to account for filtering
            batch_size = pool_size + max(50, pool_size // 100)
            random_tokens = np.random.randint(
                0, vocab_size, size=batch_size, dtype=np.int32
            )

            # Filter out special tokens
            if special_ids:
                mask = ~np.isin(random_tokens, list(special_ids))
                valid_tokens = random_tokens[mask]
            else:
                valid_tokens = random_tokens

            long_token_ids.extend(valid_tokens.tolist())

            if len(long_token_ids) >= pool_size:
                long_token_ids = long_token_ids[:pool_size]
                break

        # Take a slightly longer slice than needed (add some buffer)
        buffer_size = min(
            50, max(10, target_length // 10)
        )  # At least 10, up to 50 tokens buffer
        slice_length = min(target_length + buffer_size, len(long_token_ids))
        token_slice = long_token_ids[:slice_length]

        random.shuffle(token_slice)

        # Fine-tune to get exact target_length after tokenization
        current_tokens = token_slice[:target_length]
        prompt = tokenizer.decode(
            current_tokens, clean_up_tokenization_spaces=True
        ).strip()

        encoded_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        actual_length = len(encoded_tokens) + 1  # for the eos token
        current_tokens = encoded_tokens

        if actual_length == target_length:
            return prompt
        elif actual_length < target_length:
            # Need more tokens, add from our shuffled pool
            needed = target_length - actual_length
            if len(current_tokens) + needed <= len(token_slice):
                current_tokens = token_slice[: len(current_tokens) + needed]
            else:
                needed_from_pool = needed - (slice_length - target_length)
                current_tokens = (
                    token_slice
                    + long_token_ids[slice_length : slice_length + needed_from_pool]
                )
        else:
            # Too many tokens, remove some
            excess = actual_length - target_length
            current_tokens = current_tokens[: max(1, len(current_tokens) - excess)]

        prompt = tokenizer.decode(
            current_tokens, clean_up_tokenization_spaces=True
        ).strip()
        return prompt

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
    ) -> list[SampleRequest]:
        """Sample requests of input/output lengths from distributions."""
        # Pre-compute CDFs for sampling
        cdfs = {
            "input_tokens": np.cumsum(self.input_pdf),
            "output_tokens": np.cumsum(self.output_pdf),
        }

        requests: list[SampleRequest] = []
        logger.info(
            f"Generating {num_requests} synthetic requests with sampled lengths"
        )

        for _ in range(num_requests):
            # Sample desired input and output lengths using CDF
            random_values = self.rng.random(2)
            sampled_input_len = np.searchsorted(cdfs["input_tokens"], random_values[0])
            sampled_output_len = np.searchsorted(
                cdfs["output_tokens"], random_values[1]
            )
            sampled_input_len = max(1, sampled_input_len)
            sampled_output_len = max(1, sampled_output_len)

            # Generate random prompt and completion with target lengths
            prompt = self._generate_random_text_with_length(
                tokenizer, sampled_input_len
            )
            actual_prompt_len = len(tokenizer(prompt).input_ids)

            completion = "[This is omitted as only the output length is used]"

            requests.append(
                SampleRequest(
                    prompt=prompt,
                    completion=completion,
                    prompt_len=actual_prompt_len,
                    expected_output_len=sampled_output_len,
                    multimodal_contents=[],
                )
            )

        maybe_oversample_requests(requests, num_requests, self.random_seed)

        return requests
