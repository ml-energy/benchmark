"""Workload configurations for LLM and MLLM benchmarks.

A workload configuration defines one specific case or datapoint for benchmarking.
"""

from __future__ import annotations

import logging
from functools import cached_property
from abc import abstractmethod
from typing import Literal, Self
from pathlib import Path

import tyro
from pydantic import BaseModel, model_validator
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.fim.request import FIMRequest

from mlenergy.constants import DEFAULT_SEED
from mlenergy.llm.datasets import (
    SampleRequest,
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
        base_dir: Base directory where all workload files are stored.
            It should be unique for each workload configuration.
        seed: Random seed for reproducibility.
        model_id: Model identifier for the model to be used in the benchmark.
        num_requests: Number of requests to sample for the benchmark.
        gpu_model: GPU model identifier (e.g., "H100", "A100", "B200") used to
            load model-specific vLLM configurations.
        max_num_seqs: vLLM maximum number of seuqences config.
        max_num_batched_tokens: vLLM maximum number of batched tokens config.
        num_prefills: Number of prefill instances for disaggregated serving.
        num_decodes: Number of decode instances for disaggregated serving.
        num_prefill_warmups: Number of warmup requests for prefill steady state.
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
    num_prefills: int | None = None
    num_decodes: int | None = None
    num_prefill_warmups: int = 50

    @model_validator(mode="after")
    def _validate_workoad(self) -> Self:
        """Validate the sanity of the workload."""
        if (
            self.num_prefills is not None
            and self.num_decodes is not None
            and (self.num_prefills <= 0 or self.num_decodes <= 0)
        ):
            raise ValueError("Invalid prefills and decodes configuration")
        return self

    @cached_property
    def normalized_name(self) -> str:
        """Get a Tyro-normalized name for the workload configuration."""
        return tyro._strings.hyphen_separated_from_camel_case(self.__class__.__name__)  # type: ignore

    @property
    def endpoint_type(self) -> Literal["openai", "openai-chat"]:
        """LLM server endpoint type this workload uses."""
        return "openai-chat"

    def to_path(
        self,
        of: Literal[
            "requests", "results", "driver_log", "server_log", "multimodal_dump"
        ],
        create_dirs: bool = True,
    ) -> Path:
        """Generate a file path based on file type and workload parameters.

        Types of paths
        - requests: Path to the file where sampled requests are saved.
        - results: Path to the file where results of the benchmark are saved.
        - driver_log: Path to the file where logging outputs from the driver/client are saved.
        - server_log: Path to the file where logging outputs from the vLLM server are saved.
        - multimodal_dump: Path to the directory where multimodal data (e.g., images
            videos, audios) are dumped (when `dump_multimodal_data` is True).
        """
        dir = self.base_dir / "+".join(self.to_filename_parts())

        match of:
            case "requests":
                append = "requests.json"
            case "results":
                append = "results.json"
            case "driver_log":
                append = "driver_log.txt"
            case "server_log":
                append = "server_log.txt"
            case "multimodal_dump":
                append = "multimodal_dump"
            case _:
                raise ValueError(f"Unknown path type: {of}")

        path = dir / append
        if create_dirs:
            if not path.suffix:  # Directory
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for the model specified in the configuration."""
        if self.model_id == "mistralai/Codestral-22B-v0.1":
            return CodestralTokenizer()

        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def load_requests(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        """Load the requests from the file specified by the configuration.

        If the file does not exist, it will call `sample` to sample new requests.
        """
        path = self.to_path(of="requests")
        if not path.exists():
            logger.info("No saved requests found at %s. Sampling new requests.", path)
            requests = self.sample(dump_multimodal_data=dump_multimodal_data)
            self.save_requests(requests)
        else:
            logger.info("Loading saved requests from %s", path)
            requests = RequestsFile.model_validate_json(path.read_text()).requests
        return requests

    def save_requests(self, requests: list[SampleRequest]) -> None:
        """Save the requests to the file specified by the configuration.

        Args:
            requests: A list of SampleRequest objects to save.
        """
        path = self.to_path(of="requests")
        logger.info("Saving requests to %s", path)

        file = RequestsFile(requests=requests, workload=self)  # type: ignore
        dumped = file.model_dump_json(indent=2)
        path.write_text(dumped)

    @abstractmethod
    def to_filename_parts(self) -> list[str]:
        """Generate a list of parts that will be used to create a unique filename.

        Filename parts should be unique for each configuration given the
        same base directory (`base_dir`). It should *not* include any extensions,
        as this base filename will be extended with additional parts (e.g., `requests`,
        `results`) and file extensions (e.g., `.json`).
        """

    @abstractmethod
    def sample(self, dump_multimodal_data: bool = False) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters.

        If the requests file does not exist, this method should sample new reqeusts.
        It should *not* save the requests to disk, as this is done by `load_requests`.

        Args:
            dump_multimodal_data: If True, dump sampled multimodal data (e.g., images,
                videos, audios) to disk. Useful for data debugging.
        """


class ImageChat(WorkloadConfig):
    """Workload configuration for image chat requests."""

    num_images: int

    dataset_path: str = "lmarena-ai/VisionArena-Chat"
    dataset_split: str = "train"

    def to_filename_parts(self) -> list[str]:
        """Generate a list of parts that will be used to create a unique filename."""
        return [
            "image_chat",
            self.gpu_model,
            str(self.num_requests) + "req",
            str(self.num_images) + "image",
            str(self.seed) + "seed",
            str(self.max_num_seqs) + "max_num_seqs",
            str(self.max_num_batched_tokens) + "max_num_batched_tokens",
            *(
                [f"{self.num_prefills}p{self.num_decodes}d"]
                if self.num_prefills and self.num_decodes
                else []
            ),
        ]

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

    def to_filename_parts(self) -> list[str]:
        """Generate a list of parts that will be used to create a unique filename."""
        return [
            "video_chat",
            self.gpu_model,
            self.dataset_split,
            str(self.num_requests) + "req",
            str(self.num_videos) + "video",
            str(self.seed) + "seed",
            str(self.max_num_seqs) + "max_num_seqs",
            str(self.max_num_batched_tokens) + "max_num_batched_tokens",
            *(
                [f"{self.num_prefills}p{self.num_decodes}d"]
                if self.num_prefills and self.num_decodes
                else []
            ),
        ]

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

    def to_filename_parts(self) -> list[str]:
        """Generate a list of parts that will be used to create a unique filename."""
        return [
            "audio_chat",
            self.gpu_model,
            str(self.num_requests) + "req",
            str(self.num_audios) + "audio",
            str(self.seed) + "seed",
            str(self.max_num_seqs) + "max_num_seqs",
            str(self.max_num_batched_tokens) + "max_num_batched_tokens",
            *(
                [f"{self.num_prefills}p{self.num_decodes}d"]
                if self.num_prefills and self.num_decodes
                else []
            ),
        ]

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

    def to_filename_parts(self) -> list[str]:
        """Generate a list of parts that will be used to create a unique filename."""
        return [
            "omni_chat",
            self.gpu_model,
            str(self.num_requests) + "req",
            str(self.num_images) + "image",
            str(self.num_videos) + "video",
            str(self.num_audio) + "audio",
            str(self.seed) + "seed",
            str(self.max_num_seqs) + "max_num_seqs",
            str(self.max_num_batched_tokens) + "max_num_batched_tokens",
            *(
                [f"{self.num_prefills}p{self.num_decodes}d"]
                if self.num_prefills and self.num_decodes
                else []
            ),
        ]

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

    def to_filename_parts(self) -> list[str]:
        return [
            "lm_arena_chat",
            self.gpu_model,
            str(self.num_requests) + "req",
            str(self.seed) + "seed",
            str(self.max_num_seqs) + "max_num_seqs",
            str(self.max_num_batched_tokens) + "max_num_batched_tokens",
            *(
                [f"{self.num_prefills}p{self.num_decodes}d"]
                if self.num_prefills and self.num_decodes
                else []
            ),
        ]

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

    def to_filename_parts(self) -> list[str]:
        return [
            "length_control",
            self.gpu_model,
            str(self.num_requests) + "req",
            str(int(self.input_mean)) + "input_mean",
            str(int(self.output_mean)) + "output_mean",
            str(self.pareto_a) + "pareto_a",
            str(self.seed) + "seed",
            str(self.max_num_seqs) + "max_num_seqs",
            str(self.max_num_batched_tokens) + "max_num_batched_tokens",
            *(
                [f"{self.num_prefills}p{self.num_decodes}d"]
                if self.num_prefills and self.num_decodes
                else []
            ),
        ]

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

    def to_filename_parts(self) -> list[str]:
        return [
            "sourcegraph_fim",
            self.gpu_model,
            str(self.num_requests) + "req",
            str(self.seed) + "seed",
            str(self.max_num_seqs) + "max_num_seqs",
            str(self.max_num_batched_tokens) + "max_num_batched_tokens",
            *(
                [f"{self.num_prefills}p{self.num_decodes}d"]
                if self.num_prefills and self.num_decodes
                else []
            ),
        ]

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
    dataset_subset: str = "gpqa_extended"
    dataset_split: str = "train"

    def to_filename_parts(self) -> list[str]:
        return [
            "gpqa",
            self.gpu_model,
            str(self.num_requests) + "req",
            str(self.seed) + "seed",
            str(self.max_num_seqs) + "max_num_seqs",
            str(self.max_num_batched_tokens) + "max_num_batched_tokens",
            *(
                [f"{self.num_prefills}p{self.num_decodes}d"]
                if self.num_prefills and self.num_decodes
                else []
            ),
        ]

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
