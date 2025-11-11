"""Workload configurations for LLM and MLLM benchmarks.

A workload configuration defines one specific case or datapoint for benchmarking.
"""

from __future__ import annotations

import os
import logging
from functools import cached_property
from abc import abstractmethod
from typing import Any, Literal
from pathlib import Path

import tyro
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.fim.request import FIMRequest

from mlenergy.constants import DEFAULT_SEED
from mlenergy.llm.datasets import (
    SampleRequest,
    DataRequest,
    Tokenization,
    SourcegraphFIMDataset,
    VisionArenaDataset,
    LLaVAVideoDataset,
    AudioSkillsDataset,
    OmniDataset,
    LMArenaHumanPreferenceDataset,
    GPQADataset,
    ParetoExpDistributionDataset,
)

logger = logging.getLogger(__name__)


class CodestralTokenizer(PreTrainedTokenizer):
    """Custom tokenizer for Codestral-22B-v0.1."""

    def __init__(self) -> None:
        """Initialize the Codestral tokenizer."""
        self.name_or_path = "mistralai/Codestral-22B-v0.1"
        self.tokenizer = MistralTokenizer.from_hf_hub(self.name_or_path)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)

    def encode_fim(self, prefix: str, suffix: str) -> BatchEncoding:
        """Tokenize the input prefix and suffix."""
        fim = FIMRequest(prompt=prefix, suffix=suffix)
        tokenized = self.tokenizer.encode_fim(fim)
        return BatchEncoding(
            data={"input_ids": tokenized.tokens, "text": tokenized.text}
        )

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        """Delegate to the HuggingFace tokenizer, unless its for FIM."""
        if "prefix" in kwargs and "suffix" in kwargs:
            return self.encode_fim(kwargs["prefix"], kwargs["suffix"])
        return self.hf_tokenizer(*args, **kwargs)


class DataFile(BaseModel):
    """Wrapper model for model-independent request data.

    Attributes:
        data: A list of DataRequest objects containing model-independent data.
        dataset_params: Dataset-related parameters that uniquely identify this data.
    """

    data: list[DataRequest]
    dataset_params: dict[str, Any]


class TokenizationFile(BaseModel):
    """Wrapper model for model-dependent tokenization data.

    Attributes:
        tokenization: A list of Tokenization objects.
        model_id: The model used for tokenization.
    """

    tokenization: list[Tokenization]
    model_id: str


class RequestsFile(BaseModel):
    """Wrapper model for serializing and deserializing sampled requests.

    Attributes:
        requests: A list of SampleRequest objects.
        workload: A serialized representation of the workload configuration
            that generated these requests.

    """

    requests: list[SampleRequest]
    workload: (
        ImageChat
        | VideoChat
        | AudioChat
        | OmniChat
        | LMArenaChat
        | SourcegraphFIM
        | GPQA
        | LengthControl
    )


class WorkloadConfig(BaseModel):
    """Base class for workload configurations.

    A workload configuration defines one specific case or datapoint for benchmarking.
    It should instantiate appropriate datasets lazily and provide methods to sample
    and save requests.

    Attributes:
        base_dir: Base directory for all runs (e.g., "run/llm"). The modality and task
            are automatically appended based on the workload's properties.
        seed: Random seed for reproducibility.
        model_id: Model identifier for the model to be used in the benchmark.
        num_requests: Number of requests to sample for the benchmark.
        gpu_model: GPU model identifier (e.g., "H100", "A100", "B200") used to
            load model-specific vLLM configurations.
        max_num_seqs: vLLM maximum number of sequences config.
        max_num_batched_tokens: vLLM maximum number of batched tokens config.
    """

    # Input parameters
    base_dir: Path
    seed: int = DEFAULT_SEED
    model_id: str
    num_requests: int

    # Systems parameters
    gpu_model: str
    max_num_seqs: int
    max_num_batched_tokens: int | None = None

    @cached_property
    def normalized_name(self) -> str:
        """Get a Tyro-normalized name for the workload configuration."""
        return tyro._strings.hyphen_separated_from_camel_case(self.__class__.__name__)  # type: ignore

    @property
    def endpoint_type(self) -> Literal["openai", "openai-chat"]:
        """LLM server endpoint type this workload uses."""
        return "openai-chat"

    @property
    def use_prompt_token_ids(self) -> bool:
        """Whether to send prompt_token_ids instead of string prompts to the server.

        Most workloads send string prompts. Only specific tasks like FIM need to send
        pre-tokenized inputs.
        """
        return False

    @abstractmethod
    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters.

        If the requests file does not exist, this method should sample new reqeusts.
        It should *not* save the requests to disk, as this is done by `load_requests`.

        Args:
            dump_multimodal_data: If True, dump sampled multimodal data (e.g., images,
                videos, audios) to disk. Useful for data debugging.
        """

    @abstractmethod
    def _dataset_params(self) -> dict[str, Any]:
        """Get dataset-only parameters that determine requests.json uniqueness.

        These should include only parameters that affect the dataset sampling,
        not runtime parameters like max_num_seqs or gpu_model.
        """

    def _result_params(self) -> dict[str, Any]:
        """Get runtime parameters for results path.

        The set of parameters returned by this method should uniquely identify the
        workload configuration, as this will be used to create unique results directories.
        """
        return {
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            **self._dataset_params(),
        }

    def to_path(
        self,
        of: Literal[
            "requests",
            "tokenization",
            "multimodal_dump",
            "results",
            "driver_log",
            "server_log",
        ],
        create_dirs: bool = True,
    ) -> Path:
        """Generate a file path based on file type and workload parameters.

        Types of paths:
        - requests: Model-independent request data ({task}/requests/, shared across models)
        - multimodal_dump: Multimodal data directory ({task}/requests/, shared across models)
        - tokenization: Model-dependent tokenization ({task}/tokenization/{model_id}/)
        - results: Benchmark results ({task}/results/{model_id}/{gpu}/{runtime_params}/)
        - driver_log: Driver logs (in results dir)
        - server_log: Server logs (in results dir)
        """
        # Build task root from explicit modality and task properties
        # base_dir should be like "run/llm" or "run/mllm"
        # task_root will be like "run/llm/image-chat" or "run/mllm/video-chat"
        task_root = self.base_dir / self.normalized_name

        # Get dataset parameters for data/tokenization paths
        dataset_params = self._dataset_params()
        dataset_param_str = "+".join(f"{k}+{v}" for k, v in dataset_params.items())

        # Task-level shared data paths
        if of in ("requests", "multimodal_dump"):
            data_dir = task_root / "requests" / dataset_param_str
            match of:
                case "requests":
                    path = data_dir / "requests.json"
                case "multimodal_dump":
                    path = data_dir / "multimodal_dump"

        # Model-level tokenization path
        elif of == "tokenization":
            path = (
                task_root
                / "tokenization"
                / self.model_id
                / dataset_param_str
                / "tokenization.json"
            )

        # Results directory paths
        elif of in ("results", "driver_log", "server_log"):
            result_params = self._result_params()
            result_param_str = "+".join(f"{k}+{v}" for k, v in result_params.items())
            results_dir = (
                task_root
                / "results"
                / self.model_id
                / self.gpu_model
                / result_param_str
            )

            match of:
                case "results":
                    path = results_dir / "results.json"
                case "driver_log":
                    path = results_dir / "driver.log"
                case "server_log":
                    path = results_dir / "server.log"

            # Create symlinks to data and tokenization files when setting up results dir
            if create_dirs and not results_dir.exists():
                results_dir.mkdir(parents=True, exist_ok=True)
                self._create_data_symlinks(results_dir)

        else:
            raise ValueError(f"Unknown path type: {of}")

        if create_dirs:
            if not path.suffix:  # Directory
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _create_data_symlinks(self, results_dir: Path) -> None:
        """Create symlinks to requests and tokenization files in the results directory.

        Args:
            results_dir: The results directory where symlinks should be created.
        """
        data_path = self.to_path(of="requests", create_dirs=False)
        tokenization_path = self.to_path(of="tokenization", create_dirs=False)

        # Create relative symlinks
        data_symlink = results_dir / "requests.json"
        tokenization_symlink = results_dir / "tokenization.json"

        # Calculate relative paths from results_dir to the target files
        data_relative = os.path.relpath(data_path, results_dir)
        tokenization_relative = os.path.relpath(tokenization_path, results_dir)

        # Create symlinks (overwrite if they exist)
        if data_symlink.exists() or data_symlink.is_symlink():
            data_symlink.unlink()
        data_symlink.symlink_to(data_relative)

        if tokenization_symlink.exists() or tokenization_symlink.is_symlink():
            tokenization_symlink.unlink()
        tokenization_symlink.symlink_to(tokenization_relative)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for the model specified in the configuration."""
        if self.model_id == "mistralai/Codestral-22B-v0.1":
            return CodestralTokenizer()

        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def load_requests(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        """Load the requests from the file specified by the configuration.

        Loads model-independent data from task-level requests.json and model-dependent
        tokenization from tokenization.json, then merges them into SampleRequest objects.
        If files don't exist, samples new requests and saves them to separate files.
        """
        requests_path = self.to_path(of="requests")
        tokenization_path = self.to_path(of="tokenization")

        # Check if both new files exist
        if requests_path.exists() and tokenization_path.exists():
            logger.info("Loading data from %s", requests_path)
            logger.info("Loading tokenization from %s", tokenization_path)

            data_file = DataFile.model_validate_json(requests_path.read_text())
            tokenization_file = TokenizationFile.model_validate_json(
                tokenization_path.read_text()
            )

            # Verify model_id matches
            if tokenization_file.model_id != self.model_id:
                logger.warning(
                    "Tokenization was generated with model %s, but current model is %s. "
                    "Regenerating tokenization.",
                    tokenization_file.model_id,
                    self.model_id,
                )
                # Regenerate tokenization
                requests = self._merge_data_and_tokenization(data_file.data, None)
                self._save_tokenization(requests)
            else:
                # Merge data and tokenization
                requests = self._merge_data_and_tokenization(
                    data_file.data, tokenization_file.tokenization
                )

        elif requests_path.exists():
            # requests.json exists but tokenization doesn't
            logger.info("Loading data from %s", requests_path)
            data_file = DataFile.model_validate_json(requests_path.read_text())
            logger.info("Tokenization not found. Generating tokenization.")
            requests = self._merge_data_and_tokenization(data_file.data, None)
            self._save_tokenization(requests)

        else:
            # No files exist. Sample new requests.
            logger.info("No saved data found. Sampling new requests.")
            requests = self.sample(dump_multimodal_data=dump_multimodal_data)
            self._save_data(requests)
            self._save_tokenization(requests)

        return requests

    def _merge_data_and_tokenization(
        self,
        data_list: list[DataRequest],
        tokenization_list: list[Tokenization] | None,
    ) -> list[SampleRequest]:
        """Merge data requests with tokenization, generating tokenization if not provided."""
        if tokenization_list is None:
            # Generate tokenization from data
            logger.info("Generating tokenization using tokenizer: %s", self.model_id)
            requests = []
            for data in data_list:
                # Tokenize to get counts
                if isinstance(data.prompt, str):
                    prompt_tokens = self.tokenizer(data.prompt).input_ids
                else:
                    # Multi-turn conversation
                    prompt_tokens = []
                    for turn in data.prompt:
                        prompt_tokens.extend(self.tokenizer(turn).input_ids)

                completion_tokens = self.tokenizer(data.completion).input_ids

                tokenization = Tokenization(
                    prompt_len=len(prompt_tokens),
                    expected_output_len=len(completion_tokens),
                    prompt_token_ids=prompt_tokens,
                )
                requests.append(
                    SampleRequest.from_data_and_tokenization(data, tokenization)
                )
            return requests
        else:
            # Merge existing data and tokenization
            if len(data_list) != len(tokenization_list):
                raise ValueError(
                    f"Mismatched data and tokenization: {len(data_list)} data, {len(tokenization_list)} tokenization"
                )
            return [
                SampleRequest.from_data_and_tokenization(data, tokenization)
                for data, tokenization in zip(data_list, tokenization_list, strict=True)
            ]

    def _save_data(self, requests: list[SampleRequest]) -> None:
        """Save model-independent data to task-level requests.json."""
        data_path = self.to_path(of="requests")
        logger.info("Saving data to %s", data_path)

        data_requests = [req.to_data_request() for req in requests]
        data_file = DataFile(data=data_requests, dataset_params=self._dataset_params())
        data_path.write_text(data_file.model_dump_json(indent=2))

    def _save_tokenization(self, requests: list[SampleRequest]) -> None:
        """Save model-dependent tokenization to tokenization.json."""
        tokenization_path = self.to_path(of="tokenization")
        logger.info("Saving tokenization to %s", tokenization_path)

        tokenization_list = [req.to_tokenization() for req in requests]
        tokenization_file = TokenizationFile(
            tokenization=tokenization_list, model_id=self.model_id
        )
        tokenization_path.write_text(tokenization_file.model_dump_json(indent=2))

    def save_requests(self, requests: list[SampleRequest]) -> None:
        """Save the requests to separate requests.json and tokenization.json files.

        Args:
            requests: A list of SampleRequest objects to save.
        """
        self._save_data(requests)
        self._save_tokenization(requests)


class ImageChat(WorkloadConfig):
    """Workload configuration for image chat requests."""

    num_images: int

    dataset_path: str = "lmarena-ai/VisionArena-Chat"
    dataset_split: str = "train"

    def _dataset_params(self) -> dict[str, Any]:
        """Get dataset-only parameters."""
        return {
            "dataset_split": self.dataset_split,
            "num_requests": self.num_requests,
            "num_images": self.num_images,
            "seed": self.seed,
        }

    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters."""
        dataset = VisionArenaDataset(
            dataset_path=self.dataset_path,
            dataset_split=self.dataset_split,
            random_seed=self.seed,
        )
        requests = dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=self.num_requests,
            num_images=self.num_images,
            dump_multimodal_dir=(
                self.to_path(of="multimodal_dump") if dump_multimodal_data else None
            ),
        )
        return requests


class VideoChat(WorkloadConfig):
    """Workload configuration for video chat requests."""

    num_videos: int

    dataset_path: str = "lmms-lab/LLaVA-Video-178K"
    dataset_split: str = "caption"
    video_data_dir: str  # Uncompressed video data directory

    def _dataset_params(self) -> dict[str, Any]:
        """Get dataset-only parameters."""
        return {
            "dataset_split": self.dataset_split,
            "num_requests": self.num_requests,
            "num_videos": self.num_videos,
            "seed": self.seed,
        }

    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters."""
        dataset = LLaVAVideoDataset(
            dataset_path=self.dataset_path,
            dataset_split=self.dataset_split,
            random_seed=self.seed,
            video_data_dir=self.video_data_dir,
        )
        requests = dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=self.num_requests,
            num_videos=self.num_videos,
            dump_multimodal_dir=self.to_path(of="multimodal_dump")
            if dump_multimodal_data
            else None,
        )
        return requests


class AudioChat(WorkloadConfig):
    """Workload configuration for audio chat requests."""

    num_audios: int

    dataset_path: str = "nvidia/AudioSkills"
    dataset_split: str = "fsd50k"
    audio_data_dir: str  # Uncompressed audio data directory

    def _dataset_params(self) -> dict[str, Any]:
        """Get dataset-only parameters."""
        return {
            "dataset_split": self.dataset_split,
            "num_requests": self.num_requests,
            "num_audios": self.num_audios,
            "seed": self.seed,
        }

    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters.

        Returns:
            list[SampleRequest]: A list of SampleRequest objects sampled from
            the dataset.
        """
        dataset = AudioSkillsDataset(
            dataset_path=self.dataset_path,
            dataset_split=self.dataset_split,
            random_seed=self.seed,
            audio_data_dir=self.audio_data_dir,
        )
        requests = dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=self.num_requests,
            num_audio=self.num_audios,
            dump_multimodal_dir=(
                self.to_path(of="multimodal_dump") if dump_multimodal_data else None
            ),
        )
        return requests


class OmniChat(WorkloadConfig):
    """Workload configuration for a multi-modal dataset allows any combination of image, video, and audio data in requests."""

    num_images: int
    num_videos: int
    num_audio: int

    video_dataset: str = "lmms-lab/LLaVA-Video-178K"
    video_split: str = "caption"
    video_data_dir: str  # Uncompressed video data directory

    def _dataset_params(self) -> dict[str, Any]:
        """Get dataset-only parameters."""
        return {
            "video_dataset_split": self.video_split,
            "num_requests": self.num_requests,
            "num_images": self.num_images,
            "num_videos": self.num_videos,
            "num_audio": self.num_audio,
            "seed": self.seed,
        }

    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters."""
        dataset = OmniDataset(
            video_dataset_path=self.video_dataset,
            video_dataset_split=self.video_split,
            random_seed=self.seed,
            video_data_dir=self.video_data_dir,
        )
        requests = dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=self.num_requests,
            num_images=self.num_images,
            num_videos=self.num_videos,
            num_audio=self.num_audio,
            dump_multimodal_dir=(
                self.to_path(of="multimodal_dump") if dump_multimodal_data else None
            ),
        )
        return requests


class LMArenaChat(WorkloadConfig):
    """Workload using the LMArena human preference dataset."""

    dataset_path: str = "lmarena-ai/arena-human-preference-100k"
    dataset_split: str = "train"

    def _dataset_params(self) -> dict[str, Any]:
        """Get dataset-only parameters."""
        return {
            "dataset_split": self.dataset_split,
            "num_requests": self.num_requests,
            "seed": self.seed,
        }

    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        dataset = LMArenaHumanPreferenceDataset(
            dataset_path=self.dataset_path,
            dataset_split=self.dataset_split,
            random_seed=self.seed,
        )
        return dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=self.num_requests,
        )


class LengthControl(WorkloadConfig):
    """Workload that generates random strings with controlled input/output token lengths.

    Uses Pareto distribution for input lengths and Exponential distribution for output lengths,
    and generates synthetic random text.
    """

    input_mean: float = 500.0
    output_mean: float = 300.0
    pareto_a: float = 2.5

    def _dataset_params(self) -> dict[str, Any]:
        """Get dataset-only parameters."""
        return {
            "num_requests": self.num_requests,
            "input_mean": int(self.input_mean),
            "output_mean": int(self.output_mean),
            "pareto_a": self.pareto_a,
            "seed": self.seed,
        }

    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        dataset = ParetoExpDistributionDataset(
            input_mean=self.input_mean,
            output_mean=self.output_mean,
            pareto_a=self.pareto_a,
            random_seed=self.seed,
            model_max_length=getattr(self.tokenizer, "model_max_length", 32768),
        )
        return dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=self.num_requests,
        )


class SourcegraphFIM(WorkloadConfig):
    """Workload for the Sourcegraph FIM dataset."""

    dataset_path: str = "sourcegraph/context-aware-fim-code-completions"
    dataset_split: str = "train"

    @property
    def endpoint_type(self) -> Literal["openai", "openai-chat"]:
        """LLM server endpoint type this workload uses.

        FIM requests are pre-formatted based on the model type and the LLM is
        expected to exactly continue from the provided prompt.
        """
        return "openai"

    @property
    def use_prompt_token_ids(self) -> bool:
        """SourcegraphFIM requires sending prompt_token_ids to the server."""
        return True

    def _dataset_params(self) -> dict[str, Any]:
        """Get dataset-only parameters."""
        return {
            "dataset_split": self.dataset_split,
            "num_requests": self.num_requests,
            "seed": self.seed,
        }

    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        """Sample requests."""
        dataset = SourcegraphFIMDataset(
            dataset_path=self.dataset_path,
            dataset_split=self.dataset_split,
            random_seed=self.seed,
        )
        return dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=self.num_requests,
        )


class GPQA(WorkloadConfig):
    """Workload for the GPQA dataset."""

    dataset_path: str = "Idavidrein/gpqa"
    dataset_subset: str = "gpqa_diamond"
    dataset_split: str = "train"

    def _dataset_params(self) -> dict[str, Any]:
        """Get dataset-only parameters."""
        return {
            "dataset_subset": self.dataset_subset,
            "dataset_split": self.dataset_split,
            "num_requests": self.num_requests,
            "seed": self.seed,
        }

    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        dataset = GPQADataset(
            dataset_path=self.dataset_path,
            dataset_subset=self.dataset_subset,
            dataset_split=self.dataset_split,
            random_seed=self.seed,
        )
        return dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=self.num_requests,
        )
