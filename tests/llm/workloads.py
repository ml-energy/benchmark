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
)

logger = logging.getLogger("tests.llm.workloads")


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
        num_requests=100,
        num_images=2,
        model_id=model_id,
        gpu_model="H100",
        max_num_seqs=32,
    )
    requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    work = VideoChat(
        base_dir=Path("run/mllm/video_chat") / model_id,
        num_requests=100,
        num_videos=1,
        model_id=model_id,
        video_data_dir="/turbo/llava_video_178k",
        gpu_model="H100",
        max_num_seqs=32,
    )
    requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    work = AudioChat(
        base_dir=Path("run/mllm/audio_chat") / model_id,
        num_requests=100,
        num_audios=1,
        model_id=model_id,
        audio_data_dir="/turbo/FSD50K.dev_audio",
        gpu_model="H100",
        max_num_seqs=32,
    )
    requests = work.load_requests(dump_multimodal_data=dump_multimodal_data)
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    # work = OmniChat(
    #     base_dir=Path("run/mllm/omni") / model_id,
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
        base_dir=Path("run/llm/lmarena") / model_id,
        num_requests=100,
        model_id=model_id,
        gpu_model="H100",
        max_num_seqs=32,
    )
    requests = work.load_requests()
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )

    work = GPQA(
        base_dir=Path("run/llm/gpqa") / model_id,
        num_requests=100,
        model_id=model_id,
        gpu_model="H100",
        max_num_seqs=32,
    )
    requests = work.load_requests()
    logger.info(
        "Loaded %d requests from %s", len(requests), work.to_path(of="requests")
    )
