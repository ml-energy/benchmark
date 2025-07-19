"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation.

This file is based on https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_dataset.py
"""

from abc import ABC, abstractmethod
import base64
import io
from io import BytesIO
import json
import json
import logging
import os
from pathlib import Path
import pathlib
import random
from typing import Any, Callable, Literal

from PIL import Image
from datasets import (
    get_dataset_config_names,
    get_dataset_split_names,
    interleave_datasets,
    load_dataset,
)
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s[%(name)s:%(lineno)d] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_ROOT = "workload"

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


class SampleRequest(BaseModel):
    """Represents a single inference request for benchmarking.

    Args:
        prompt: The input text prompt for the model.
    """

    prompt: str | Any
    completion: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: list[dict[str, Any]]
    """ A list of multimodal content dictionaries. """
    multi_modal_data_ids: list[str]
    """ A list of paths or identifiers for each multimodal data. """


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0

    def __init__(
        self,
        dataset_path: str,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """Initialize the BenchmarkDataset with an optional dataset path and random

        seed.  Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.data = None

    def load_data(self) -> None:
        """Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError("load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(
        self, tokenizer: PreTrainedTokenizerBase, num_requests: int
    ) -> list[SampleRequest]:
        """Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(
        self, requests: list[SampleRequest], num_requests: int
    ) -> None:
        """Oversamples the list of requests if its size is less than the desired number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            logger.warning(
                "Oversampling requests to reach %d total samples.", num_requests
            )
            random.seed(self.random_seed)
            additional = random.choices(requests, k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.", num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (
        prompt_too_short or output_too_short or prompt_too_long or combined_too_long
    )


def process_audio_bytes(data: bytes | io.BytesIO) -> dict[str, Any]:
    """Process raw audio bytes and return a multimodal content dictionary.

    Args:
        data (bytes): Raw audio data as bytes.

    Returns:
        dict[str, Any]: A dictionary containing the base64-encoded audio URL.
    """
    if isinstance(data, io.BytesIO):
        data = data.getvalue()
    audio_base64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "audio_url",
        "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
    }


def process_video_bytes(data: bytes) -> dict[str, Any]:
    """Process raw video bytes and return a multimodal content dictionary.

    Args:
        data (bytes): Raw video data as bytes.

    Returns:
        dict[str, Any]: A dictionary containing the base64-encoded video URL.
    """
    video_base64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
    }


def process_image(image: Any) -> dict[str, Any]:
    """Process a single image input and return a multimedia content dictionary.

    Supports three input types:

    1. Dictionary with raw image bytes: - Expects a dict with a 'bytes' key
       containing raw image data.  - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input: - Converts the image to RGB.  - Saves the image as
       a JPEG in memory.  - Encodes the JPEG data as a base64 string.  - Returns
       a dictionary with the image as a base64 data URL.

    3. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(BytesIO(image["bytes"]))
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
        f"Invalid image input {image}. Must be a PIL.Image.Image"
        " or str or dictionary with raw image bytes."
    )


# -----------------------------------------------------------------------------
# HuggingFace Dataset Base Implementation
# -----------------------------------------------------------------------------
class HuggingFaceDataset(BenchmarkDataset):
    """Base class for datasets hosted on HuggingFace."""

    SUPPORTED_DATASET_PATHS: set[str] | dict[str, Callable] = set()

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        dataset_subset: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        self.data = self.data.shuffle(seed=self.random_seed)


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(HuggingFaceDataset):
    """Vision Arena Dataset."""

    SUPPORTED_DATASET_PATHS = {
        "lmarena-ai/VisionArena-Chat": lambda x: x["conversation"][0][0]["content"],
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int | None = None,
        num_images: int = 1,
        save_mm_data: bool = True,
    ) -> list[SampleRequest]:
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests * num_images:
                break
            parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.dataset_path)  # type: ignore
            if parser_fn is None:
                raise ValueError(f"Unsupported dataset path: {self.dataset_path}")
            prompt = parser_fn(item)
            mm_data_id = item["images"][0]["path"]  # type: ignore
            # here id is path filename
            if save_mm_data:
                mm_data_path = Path(DATA_ROOT) / "images" / mm_data_id
                mm_data_path.parent.mkdir(parents=True, exist_ok=True)
                if not mm_data_path.exists():
                    with open(mm_data_path, "wb") as f:
                        f.write(item["images"][0]["bytes"])  # type: ignore
            mm_content = process_image(item["images"][0])  # type: ignore
            model_response = item["conversation"][1][0]["content"]  # type: ignore
            prompt_len = len(tokenizer(prompt).input_ids)
            expected_output_len = len(tokenizer(model_response).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    completion=model_response,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len
                    if output_len is None
                    else output_len,
                    multi_modal_data=[mm_content],
                    multi_modal_data_ids=[mm_data_id],
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests * num_images)
        # merge requests if num_images > 1
        if num_images > 1:
            merged_requests = []
            for i in range(0, len(sampled_requests), num_images):
                merging_requests = sampled_requests[i : i + num_images]
                merged_request = SampleRequest(
                    prompt="\n".join(r.prompt for r in merging_requests),
                    completion="\n".join(r.completion for r in merging_requests),
                    prompt_len=sum(r.prompt_len for r in merging_requests),
                    expected_output_len=sum(
                        r.expected_output_len for r in merging_requests
                    ),
                    multi_modal_data=[
                        item for r in merging_requests for item in r.multi_modal_data
                    ],
                    multi_modal_data_ids=[
                        id for r in merging_requests for id in r.multi_modal_data_ids
                    ],
                )
                merged_requests.append(merged_request)
        return sampled_requests if num_images == 1 else merged_requests


class OmniDataset(BenchmarkDataset):
    """Dataset with Video, Audio, Image, and Text."""

    def __init__(
        self,
        image_dataset_path: str,
        image_dataset_split: str,
        video_dataset_path: str,
        video_dataset_split: str,
        audio_dataset_path: str,
        audio_dataset_split: str,
        random_seed: int = 0,
    ) -> None:
        super().__init__(dataset_path="synthesized", random_seed=random_seed)
        self.image_dataset_path = image_dataset_path
        self.image_dataset_split = image_dataset_split
        self.video_dataset_path = video_dataset_path
        self.video_dataset_split = video_dataset_split
        self.audio_dataset_path = audio_dataset_path
        self.audio_dataset_split = audio_dataset_split
        self.load_data()

    def load_data(self) -> None:
        """Load data from the Image, Video, Audio, datasets respectively."""
        self.image_dataset = VisionArenaDataset(
            dataset_path=self.image_dataset_path,
            dataset_split=self.image_dataset_split,
            random_seed=self.random_seed,
        )
        self.video_dataset = LLaVAOVDataset(
            dataset_path=self.video_dataset_path,
            dataset_split=self.video_dataset_split,
            random_seed=self.random_seed,
        )
        self.audio_dataset = AudioSkillsDataset(
            dataset_path=self.audio_dataset_path,
            dataset_split=self.audio_dataset_split,
            random_seed=self.random_seed,
        )

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int | None = None,
        num_images: int = 1,
        num_videos: int = 1,
        num_audio: int = 1,
        save_mm_data: bool = True,
    ) -> list[SampleRequest]:
        sampled_image_requests = (
            self.image_dataset.sample(
                tokenizer,
                num_requests=num_requests,
                output_len=output_len,
                num_images=num_images,
                save_mm_data=save_mm_data,
            )
            if num_images > 0
            else []
        )
        sampled_video_requests = (
            self.video_dataset.sample(
                tokenizer,
                num_requests=num_requests,
                output_len=output_len,
                num_videos=num_videos,
            )
            if num_videos > 0
            else []
        )
        sampled_audio_requests = (
            self.audio_dataset.sample(
                tokenizer,
                num_requests=num_requests,
                output_len=output_len,
                num_audio=num_audio,
                save_mm_data=save_mm_data,
            )
            if num_audio > 0
            else []
        )
        merged_requests = []
        for i in range(num_requests):
            merging_requests = []
            if num_images > 0:
                merging_requests.append(sampled_image_requests[i])
            if num_videos > 0:
                merging_requests.append(sampled_video_requests[i])
            if num_audio > 0:
                merging_requests.append(sampled_audio_requests[i])
            new_request = SampleRequest(
                prompt="\n".join(r.prompt for r in merging_requests),
                completion="\n".join(r.completion for r in merging_requests),
                prompt_len=sum(r.prompt_len for r in merging_requests),
                expected_output_len=sum(
                    r.expected_output_len for r in merging_requests
                ),
                multi_modal_data=[
                    item for r in merging_requests for item in r.multi_modal_data
                ],
                multi_modal_data_ids=[
                    id for r in merging_requests for id in r.multi_modal_data_ids
                ],
            )
            merged_requests.append(new_request)

        self.maybe_oversample_requests(merged_requests, num_requests)
        return merged_requests


class AudioSkillsDataset(HuggingFaceDataset):
    """Dataset of nvidia/AudioSkills for Audio-text-to-text."""

    SUPPORTED_DATASET_PATHS = {"nvidia/AudioSkills"}

    """
    The FSD50K dataset is at https://zenodo.org/records/4060432
    Note there is Huggingface mirror but you might encounter rate limits.
    To donwload: 
    ```
    pip install -q zenodo-get
    zenodo_get 10.5281/zenodo.4060432
    ```
    """
    FSD_50K_PATHS = [Path(DATA_ROOT) / "fsd50k_audio"]

    def _get_audio_bytes(self, filename: str) -> bytes | None:
        """Return the audio bytes from the dataset item."""
        for path in self.FSD_50K_PATHS:
            full_path = pathlib.Path(path) / filename
            if full_path.exists():
                with open(full_path, "rb") as f:
                    return f.read()
        return None

    def _get_saved_audio_bytes(self, path: Path) -> bytes:
        assert path.exists(), f"Audio file {path} does not exist."
        return path.read_bytes()

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int | None = None,
        num_audio: int = 1,
        save_mm_data: bool = True,
    ) -> list[SampleRequest]:
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests * num_audio:
                break
            mm_data_id = item["sound"]  # type: ignore
            data_filename = Path(DATA_ROOT) / "audio" / mm_data_id
            if data_filename.exists():
                mm_contect_bytes = self._get_saved_audio_bytes(data_filename)
            else:
                mm_contect_bytes = self._get_audio_bytes(item["sound"])  # type: ignore
            if not mm_contect_bytes:
                logger.warning(
                    "Skipping item with missing audio file %s", item["sound"]
                )  # type: ignore
                continue
            mm_content = process_audio_bytes(mm_contect_bytes)
            if save_mm_data:
                data_filename.parent.mkdir(parents=True, exist_ok=True)
                with open(data_filename, "wb") as f:
                    f.write(mm_contect_bytes)
            conversations = item["conversations"]  # type: ignore
            prompt, completion = conversations[0]["value"], conversations[1]["value"]
            prompt_len = len(tokenizer(prompt).input_ids)
            expected_output_len = len(tokenizer(completion).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    completion=completion,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len
                    if output_len is None
                    else output_len,
                    multi_modal_data=[mm_content],
                    multi_modal_data_ids=[mm_data_id],
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests * num_audio)
        if num_audio:
            merged_requests = []
            for i in range(0, len(sampled_requests), num_audio):
                merging_requests = sampled_requests[i : i + num_audio]
                merged_request = SampleRequest(
                    prompt="\n".join(r.prompt for r in merging_requests),
                    completion="\n".join(r.completion for r in merging_requests),
                    prompt_len=sum(r.prompt_len for r in merging_requests),
                    expected_output_len=sum(
                        r.expected_output_len for r in merging_requests
                    ),
                    multi_modal_data=[
                        item for r in merging_requests for item in r.multi_modal_data
                    ],
                    multi_modal_data_ids=[
                        id for r in merging_requests for id in r.multi_modal_data_ids
                    ],
                )
                merged_requests.append(merged_request)
        return sampled_requests if num_audio == 1 else merged_requests


# -----------------------------------------------------------------------------
# Conversation Dataset Implementation
# -----------------------------------------------------------------------------


class LLaVAOVDataset(HuggingFaceDataset):
    """Dataset for LLaVA-OneVision data."""

    SUPPORTED_DATASET_PATHS = {"lmms-lab/LLaVA-Video-178K"}

    """
    This is meant to be the external storage for all the videos in the dataset.
    The videos are stored in tar.gz files, so you need download and extract them.
    The entire dataset is about 1.2TB, and you need twice that space to extract.
    ```
    huggingface-cli download lmms-lab/LLaVA-Video-178K \
        --repo-type dataset \
        --include "*.tar.gz" \
        --resume-download

    python3 prepare_llava_videos.py workload/llava_videos --jobs 12
    ```
    """
    EXTRACTED_VIDEO_PATH = Path(DATA_ROOT) / "llava_videos"

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        streams = []
        for subset in get_dataset_config_names(self.dataset_path):
            if self.dataset_split in get_dataset_split_names(
                self.dataset_path,
                config_name=subset,
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
                continue
        self.data = interleave_datasets(streams, seed=self.random_seed)  # type: ignore
        self.data = self.data.shuffle(seed=self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int | None = None,
        num_videos: int = 1,
        save_mm_data: bool = True,
    ) -> list[SampleRequest]:
        """Sample requests from the dataset."""
        # Filter examples with at least 2 conversations
        filtered_data = self.data.filter(lambda x: len(x["conversations"]) >= 2)
        sampled_requests = []
        dynamic_output = output_len is None
        for item in filtered_data:
            if len(sampled_requests) >= num_requests * num_videos:
                break
            if "video" not in item or not item["video"]:  # type: ignore
                logger.warning(f"Skipping item {item['id']} with missing video path.")  # type: ignore
                continue
            mm_data_id = item["video"]  # type: ignore
            data_filename = Path(DATA_ROOT) / "videos" / mm_data_id
            if data_filename.exists():
                mm_content = process_video_bytes(data_filename.read_bytes())
            else:
                extracted_path = self.EXTRACTED_VIDEO_PATH / mm_data_id
                if not extracted_path.exists():
                    logger.warning(
                        f"Video file path {str(extracted_path)} does not exist. Skipping item."
                    )  # type: ignore
                    continue
                mm_content = process_video_bytes(extracted_path.read_bytes())
                if save_mm_data:
                    # save the video file to the data root
                    data_filename.parent.mkdir(parents=True, exist_ok=True)
                    with open(data_filename, "wb") as f:
                        f.write(extracted_path.read_bytes())

            conv = item["conversations"]  # type: ignore
            prompt, completion = conv[0]["value"], conv[1]["value"]
            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len, completion_len):
                continue
            mm_data_id = item["video"]  # type: ignore
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    completion=completion,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=[mm_content],
                    multi_modal_data_ids=[mm_data_id],
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests * num_videos)
        if num_videos > 1:
            merged_requests = []
            for i in range(0, len(sampled_requests), num_videos):
                merging_requests = sampled_requests[i : i + num_videos]
                merged_request = SampleRequest(
                    prompt="\n".join(r.prompt for r in merging_requests),
                    completion="\n".join(r.completion for r in merging_requests),
                    prompt_len=sum(r.prompt_len for r in merging_requests),
                    expected_output_len=sum(
                        r.expected_output_len for r in merging_requests
                    ),
                    multi_modal_data=[
                        item for r in merging_requests for item in r.multi_modal_data
                    ],
                    multi_modal_data_ids=[
                        id for r in merging_requests for id in r.multi_modal_data_ids
                    ],
                )
                merged_requests.append(merged_request)
        return sampled_requests if num_videos == 1 else merged_requests


class WorkloadConfig(BaseModel):
    """Base class for workload configurations.

    This class allows saving and loading sample requests from a file based on
    the configuration parameters.
    """

    def to_path(self) -> Path:
        """Generate a file path based on the configuration parameters."""
        if not Path(DATA_ROOT).exists():
            os.makedirs(DATA_ROOT, exist_ok=True)
        return Path(DATA_ROOT) / self.to_filename()

    def load(self) -> list[SampleRequest]:
        """Load the requests from the file specified by the configuration.

        Returns:
            list[SampleRequest]: A list of SampleRequest objects loaded from
            the specified file.
        """
        path = self.to_path()
        if not path.exists():
            logger.info("No saved requests found at %s. Sampling new requests.", path)
            return self.sample()
        with path.open("r") as f:
            logger.info("Loading saved requests from %s", path)
            data = json.load(f)
            return [SampleRequest.model_validate_json(request) for request in data]

    def save(self, requests: list[SampleRequest]) -> None:
        """Save the requests to the file specified by the configuration.

        Args:
            requests (list[SampleRequest]): A list of SampleRequest objects to
            save.
        """
        path = self.to_path()
        with path.open("w") as f:
            json.dump([request.model_dump_json() for request in requests], f, indent=2)

    @abstractmethod
    def to_filename(self) -> str:
        """Generate a filename based on the configuration parameters."""

    @abstractmethod
    def sample(self) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters.

        This method should generate new requests if the file does not exist.
        """


class ImageWorkloadConfig(WorkloadConfig):
    """Workload configuration for image-based requests."""

    num_requests: int
    num_images: int
    seed: int = 48105

    model_id: str
    dataset: Literal["lmarena-ai/VisionArena-Chat"] = "lmarena-ai/VisionArena-Chat"

    def to_filename(self) -> str:
        """Generate a filename based on the configuration parameters.

        Returns:
            str: A filename string that includes the number of requests,
            number of images, and seed.
        """
        return f"image+{self.num_requests}reqs+{self.num_images}images+{self.seed}seed.json"

    def sample(self, save_mm_data=True) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters.

        Returns:
            list[SampleRequest]: A list of SampleRequest objects sampled from
            the dataset.
        """
        dataset = VisionArenaDataset(
            dataset_path=self.dataset,
            random_seed=self.seed,
            dataset_subset=None,
            dataset_split="train",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        requests = dataset.sample(
            tokenizer,
            num_requests=self.num_requests,
            num_images=self.num_images,
            save_mm_data=save_mm_data,
        )
        self.save(requests)
        return requests


class VideoworkloadConfig(WorkloadConfig):
    """Workload configuration for video-based requests."""

    num_requests: int
    num_videos: int
    seed: int = 48105

    model_id: str
    dataset: Literal["lmms-lab/LLaVA-Video-178K"] = "lmms-lab/LLaVA-Video-178K"
    split: str = "caption"

    def to_filename(self) -> str:
        """Generate a filename based on the configuration parameters."""
        return (
            f"video+{self.split}+"
            f"{self.num_requests}reqs+{self.num_videos}videos+{self.seed}seed.json"
        )

    def sample(self, save_mm_data=True) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters.

        Returns:
            list[SampleRequest]: A list of SampleRequest objects sampled from
            the dataset.
        """
        dataset = LLaVAOVDataset(
            dataset_path=self.dataset,
            dataset_split=self.split,
            random_seed=self.seed,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        requests = dataset.sample(
            tokenizer,
            num_requests=self.num_requests,
            num_videos=self.num_videos,
            save_mm_data=save_mm_data,
        )
        self.save(requests)
        return requests


class AudioWorkloadConfig(WorkloadConfig):
    """Workload configuration for audio-based requests."""

    num_requests: int
    num_audio: int

    seed: int = 48105
    model_id: str

    dataset: Literal["nvidia/AudioSkills"] = "nvidia/AudioSkills"
    split: str = "fsd50k"

    def to_filename(self) -> str:
        """Generate a filename based on the configuration parameters."""
        return (
            f"audio+{self.num_requests}reqs+{self.num_audio}audio+{self.seed}seed.json"
        )

    def sample(self, save_mm_data=True) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters.

        Returns:
            list[SampleRequest]: A list of SampleRequest objects sampled from
            the dataset.
        """
        dataset = AudioSkillsDataset(
            dataset_path="nvidia/AudioSkills",
            random_seed=self.seed,
            dataset_split=self.split,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        requests = dataset.sample(
            tokenizer,
            num_requests=self.num_requests,
            num_audio=self.num_audio,
            save_mm_data=save_mm_data,
        )
        self.save(requests)
        return requests


class OmniWorkloadConfig(WorkloadConfig):
    """Workload configuration for a multi-modal dataset allows any combination of image, video, and audio data in requests."""

    num_requests: int
    num_images: int
    num_videos: int
    num_audio: int

    seed: int = 48105
    model_id: str

    image_dataset: Literal["lmarena-ai/VisionArena-Chat"] = (
        "lmarena-ai/VisionArena-Chat"
    )
    image_split: Literal["train"] = "train"

    video_dataset: Literal["lmms-lab/LLaVA-Video-178K"] = "lmms-lab/LLaVA-Video-178K"
    video_split: str = "caption"

    audio_dataset: Literal["nvidia/AudioSkills"] = "nvidia/AudioSkills"
    audio_split: Literal["fsd50k"] = "fsd50k"

    def to_filename(self) -> str:
        """Generate a filename based on the configuration parameters."""
        return (
            f"omni+{self.num_requests}reqs+{self.num_images}images+"
            f"{self.num_videos}videos+{self.num_audio}audio+{self.seed}seed.json"
        )

    def sample(self, save_mm_data=True) -> list[SampleRequest]:
        """Sample requests based on the configuration parameters.

        Returns:
            list[SampleRequest]: A list of SampleRequest objects sampled from
            the dataset.
        """
        dataset = OmniDataset(
            image_dataset_path=self.image_dataset,
            image_dataset_split=self.image_split,
            video_dataset_path=self.video_dataset,
            video_dataset_split=self.video_split,
            audio_dataset_path=self.audio_dataset,
            audio_dataset_split=self.audio_split,
            random_seed=self.seed,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        requests = dataset.sample(
            tokenizer,
            num_requests=self.num_requests,
            num_images=self.num_images,
            num_videos=self.num_videos,
            num_audio=self.num_audio,
            save_mm_data=save_mm_data,
        )
        self.save(requests)
        return requests


if __name__ == "__main__":
    # Example usage
    model_id = "Qwen/Qwen2.5-Omni-7B"

    config = ImageWorkloadConfig(
        num_requests=20,
        num_images=2,
        seed=48105,
        model_id=model_id,
    )
    requests = config.load()
    print(f"Loaded {len(requests)} requests from {config.to_path()}")

    video_config = VideoworkloadConfig(
        num_requests=30,
        num_videos=1,
        seed=48105,
        model_id=model_id,
        split="caption",
    )

    requests = video_config.load()
    print(f"Loaded {len(requests)} requests from {video_config.to_path()}")

    audio_config = AudioWorkloadConfig(
        num_requests=40,
        num_audio=1,
        seed=48105,
        model_id=model_id,
    )

    audio_requests = audio_config.load()
    print(f"Loaded {len(audio_requests)} requests from {audio_config.to_path()}")

    omni_config = OmniWorkloadConfig(
        num_requests=10,
        num_images=1,
        num_videos=1,
        num_audio=2,
        seed=48105,
        model_id=model_id,
    )
    omni_requests = omni_config.load()
    print(f"Loaded {len(omni_requests)} requests from {omni_config.to_path()}")

    pass
