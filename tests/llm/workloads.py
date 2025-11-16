"""Tests for LLM/MLLM workloads."""

from __future__ import annotations

import logging
from pathlib import Path

from mlenergy.llm.workloads import (
    ImageChat,
    VideoChat,
    AudioChat,
    LMArenaChat,
    GPQA,
    SourcegraphFIM,
)

logger = logging.getLogger("tests.llm.workloads")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )

    dump_multimodal_data = True
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    work = ImageChat(
        base_dir=Path("test_run/mllm"),
        num_requests=100,
        num_images=2,
        model_id=model_id,
        gpu_model="H100",
        num_gpus=1,
        max_num_seqs=32,
    )
    requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    work = VideoChat(
        base_dir=Path("test_run/mllm"),
        num_requests=100,
        num_videos=1,
        model_id=model_id,
        video_data_dir="/turbo/llava_video_178k",
        gpu_model="H100",
        num_gpus=1,
        max_num_seqs=32,
    )
    requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    work = AudioChat(
        base_dir=Path("test_run/mllm"),
        num_requests=100,
        num_audios=1,
        model_id=model_id,
        audio_data_dir="/turbo/FSD50K.dev_audio",
        gpu_model="H100",
        num_gpus=1,
        max_num_seqs=32,
    )
    requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    # work = OmniChat(
    #     base_dir=Path("test_run/mllm"),
    #     num_requests=10,
    #     num_images=1,
    #     num_videos=1,
    #     num_audio=2,
    #     model_id=model_id,
    #     gpu_model="H100",
    #     video_data_dir="/turbo/llava_video_178k",
    # max_num_seqs=512,
    # )
    # omni_requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    # print(f"Loaded {len(omni_requests)} requests from {work.to_path(of='requests')}")

    work = LMArenaChat(
        base_dir=Path("test_run/llm"),
        num_requests=100,
        model_id=model_id,
        gpu_model="H100",
        num_gpus=1,
        max_num_seqs=32,
    )
    requests = work.load_requests()
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    # Different max_num_seqs
    work = LMArenaChat(
        base_dir=Path("test_run/llm"),
        num_requests=100,
        model_id=model_id,
        gpu_model="H100",
        num_gpus=1,
        max_num_seqs=64,
    )
    requests = work.load_requests()
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    # Different model
    work = LMArenaChat(
        base_dir=Path("test_run/llm"),
        num_requests=100,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_model="H100",
        num_gpus=1,
        max_num_seqs=32,
    )
    requests = work.load_requests()
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    work = GPQA(
        base_dir=Path("test_run/llm"),
        num_requests=100,
        model_id=model_id,
        gpu_model="H100",
        num_gpus=1,
        max_num_seqs=32,
    )
    requests = work.load_requests()
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    model_id = "mistralai/Codestral-22B-v0.1"
    work = SourcegraphFIM(
        base_dir=Path("test_run/llm"),
        num_requests=100,
        model_id=model_id,
        gpu_model="H100",
        num_gpus=1,
        max_num_seqs=32,
    )
    requests = work.load_requests()
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )
