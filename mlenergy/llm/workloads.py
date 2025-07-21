"""Workload configurations for LLM and MLLM benchmarks.

A workload configuration defines one specific case or datapoint for benchmarking.
"""

from __future__ import annotations

import logging
from functools import cached_property
from abc import abstractmethod
from typing import Literal, TYPE_CHECKING
from pathlib import Path

from pydantic import BaseModel
from transformers import AutoTokenizer

from mlenergy.constants import DEFAULT_SEED
from mlenergy.llm.datasets import (
    SampleRequest,
    VisionArenaDataset,
    LLaVAVideoDataset,
    AudioSkillsDataset,
    OmniDataset,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class RequestsFile(BaseModel):
    """Wrapper model for serializing and deserializing sampled requests.

    Attributes:
        requests: A list of SampleRequest objects.
        workload: A serialized representation of the workload configuration
            that generated these requests.

    """

    requests: list[SampleRequest]
    workload: (
        ImageChat | VideoChat | AudioChat | OmniChat
    )


class WorkloadConfig(BaseModel):
    """Base class for workload configurations.

    A workload configuration defines one specific case or datapoint for benchmarking.
    It should instantiate appropriate datasets lazily and provide methods to sample
    and save requests.
    """

    base_dir: Path
    seed: int = DEFAULT_SEED
    model_id: str
    num_requests: int

    def to_path(
        self, of: Literal["requests", "results", "driver_log", "server_log", "multimodal_dump"]
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
        if not path.suffix:  # Directory
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for the model specified in the configuration."""
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
            str(self.num_requests) + "req",
            str(self.num_images) + "image",
            str(self.seed) + "seed",
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
            self.dataset_split,
            str(self.num_requests) + "req",
            str(self.num_videos) + "video",
            str(self.seed) + "seed",
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
            str(self.num_requests) + "req",
            str(self.num_audios) + "audio",
            str(self.seed) + "seed",
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
            str(self.num_requests) + "req",
            str(self.num_images) + "image",
            str(self.num_videos) + "video",
            str(self.num_audio) + "audio",
            str(self.seed) + "seed",
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s[%(name)s:%(lineno)d] - %(message)s",
        datefmt="%H:%M:%S",
    )

    dump_multimodal_data = True

    model_id = "Qwen/Qwen2.5-Omni-7B"

    work = ImageChat(
        base_dir=Path("run/mllm/image_chat") / model_id,
        num_requests=30,
        num_images=2,
        model_id=model_id,
    )
    requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    work = VideoChat(
        base_dir=Path("run/mllm/video_chat") / model_id,
        num_requests=30,
        num_videos=1,
        model_id=model_id,
        video_data_dir="/turbo/llava_video_178k",
    )

    requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    work = AudioChat(
        base_dir=Path("run/mllm/audio_chat") / model_id,
        num_requests=40,
        num_audios=1,
        model_id=model_id,
        audio_data_dir="/turbo/FSD50K.dev_audio",
    )

    audio_requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    logger.info(
        "Loaded %d requests from %s", len(audio_requests), work.to_path(of="requests")
    )

    # work = OmniChatWorkload(
    #     base_dir=Path("run/mllm/omni") / model_id,
    #     num_requests=10,
    #     num_images=1,
    #     num_videos=1,
    #     num_audio=2,
    #     model_id=model_id,
    #     video_data_dir="/turbo/llava_video_178k",
    # )
    # omni_requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    # print(f"Loaded {len(omni_requests)} requests from {work.to_path(of='requests')}")
