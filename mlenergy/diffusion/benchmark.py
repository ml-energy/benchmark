"""Diffusion model benchmark runner.

Similar to LLM benchmark, this provides comprehensive benchmarking
for various diffusion models with energy monitoring and detailed metrics.
"""

from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import os
import random
import sys
import time
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import torch
import torch.distributed
import tyro
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers.models.t5 import T5EncoderModel
from zeus.monitor import ZeusMonitor, PowerMonitor, TemperatureMonitor
from zeus.show_env import show_env
import math
import functools
from typing import Optional, Dict, Union

from xfuser import (
    xFuserArgs,
    xFuserFluxPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserHunyuanDiTPipeline,
    xFuserSanaPipeline,
    xFuserLattePipeline,
)
from xfuser import (
    xFuserCogVideoXPipeline,
    xFuserConsisIDPipeline,
)
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_sp_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    initialize_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_pipeline_parallel_world_size,
)
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.pipelines.consisid.consisid_utils import (
    prepare_face_models,
    process_face_embeddings_infer,
)
from diffusers import WanPipeline, WanImageToVideoPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanAttnProcessor
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND

from mlenergy.diffusion.dataset import DiffusionRequest
from mlenergy.diffusion.workloads import (
    DiffusionWorkloadConfig,
    TextToImage,
    TextToVideo,
)

logger = logging.getLogger("mlenergy.diffusion.run")

WorkloadT = TypeVar("WorkloadT", bound=DiffusionWorkloadConfig)

# Pipeline configurations - map model_id to pipeline
PIPELINE_CONFIGS = {
    "black-forest-labs/FLUX.1-dev": {
        "pipeline_class": xFuserFluxPipeline,
        "needs_t5": True,
        "t5_subfolder": "text_encoder_2",
        "dtype": torch.bfloat16,
    },
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": {
        "pipeline_class": xFuserPixArtSigmaPipeline,
        "needs_t5": True,
        "t5_subfolder": "text_encoder",
        "dtype": torch.bfloat16,
    },
    "stabilityai/stable-diffusion-3-medium-diffusers": {
        "pipeline_class": xFuserStableDiffusion3Pipeline,
        "needs_t5": True,
        "t5_subfolder": "text_encoder_3",
        "dtype": torch.bfloat16,
    },
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers": {
        "pipeline_class": xFuserHunyuanDiTPipeline,
        "needs_t5": True,
        "t5_subfolder": "text_encoder_2",
        "dtype": torch.bfloat16,
    },
    "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers": {
        "pipeline_class": xFuserSanaPipeline,
        "needs_t5": False,
        "dtype": torch.bfloat16,
    },
    "zai-org/CogVideoX1.5-5B": {
        "pipeline_class": xFuserCogVideoXPipeline,
        "needs_t5": False,  # CogVideoX uses built-in text encoder, not T5
        "dtype": torch.bfloat16,
    },
    # ConsisID official repo id; also support local directories detected dynamically
    "BestWishYsh/ConsisID-preview": {
        "pipeline_class": xFuserConsisIDPipeline,
        "needs_t5": False,
        "dtype": torch.bfloat16,
    },
    "maxin-cn/Latte-1": {
        "pipeline_class": xFuserLattePipeline,
        "needs_t5": False,
        "dtype": torch.bfloat16,
    },
    # Wan 2.1 uses diffusers pipeline directly with xFuser runtime integration
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": {
        "pipeline_class": None,
        "needs_t5": False,
        "dtype": torch.bfloat16,
    },
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {
        "pipeline_class": None,
        "needs_t5": False,
        "dtype": torch.bfloat16,
    },
    # HunyuanVideo uses diffusers pipeline with xFuser runtime integration
    "tencent/HunyuanVideo": {
        "pipeline_class": None,
        "needs_t5": False,
        "dtype": torch.bfloat16,  # follow official example
    },
}


class DiffusionArgs(BaseModel, Generic[WorkloadT]):
    """Data model for diffusion benchmark arguments.

    Attributes:
        workload: Workload configuration for the benchmark.
        warmup_iters: Number of warmup iterations before benchmarking.
        benchmark_iters: Number of benchmark iterations.
        overwrite_results: Whether to overwrite existing results.
        save_images: Whether to save generated images.
    """

    # Workload configuration
    workload: WorkloadT
    warmup_iters: int = 2
    benchmark_iters: int = 4

    # Results configuration
    overwrite_results: bool = False
    save_images: bool = True


def get_model_type_from_id(model_id: str) -> str:
    """Get model type from model_id for backward compatibility."""
    if "FLUX" in model_id:
        return "Flux"
    elif "stable-diffusion-xl" in model_id:
        return "SDXL"
    elif "PixArt-alpha" in model_id and "Sigma" not in model_id:
        return "Pixart-alpha"
    elif "PixArt-Sigma" in model_id:
        return "Pixart-sigma"
    elif "stable-diffusion-3" in model_id:
        return "Sd3"
    elif "HunyuanDiT" in model_id:
        return "HunyuanDiT"
    elif "SANA" in model_id or "Sana" in model_id:
        return "Sana"
    elif "CogVideo" in model_id:
        return "CogVideo"
    elif "ConsisID" in model_id or "ConsisID-preview" in model_id:
        return "ConsisID"
    elif "Latte" in model_id or "latte" in model_id:
        return "Latte"
    elif "Wan" in model_id:
        return "Wan"
    elif "HunyuanVideo" in model_id:
        return "HunyuanVideo"
    else:
        return "Unknown"


def _parallelize_wan_transformer(pipe: Any) -> None:
    """Patch Wan pipeline transformer for sequence-parallel attention using xFuserWanAttnProcessor."""
    transformer = pipe.transformer
    transformer_2 = getattr(pipe, "transformer_2", None)

    def maybe_transformer_2(t2):
        if t2 is not None:
            return functools.wraps(t2.__class__.forward)
        else:
            return (lambda f: f)

    @functools.wraps(transformer.__class__.forward)
    @maybe_transformer_2(transformer_2)
    def new_forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=True,
        attention_kwargs=None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            _ = attention_kwargs.pop("scale", 1.0)  # lora scale, unused

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))  # batch_size, seq_len, 6, inner_dim
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))  # batch_size, 6, inner_dim

        if encoder_hidden_states_image is not None:
            # Wan2.1: when doing cross attention with image embeddings
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        else:
            # Chunk EHS across sequence-parallel groups
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2
            )[get_sequence_parallel_rank()]

        # Sequence-parallel: pad to multiple of sp world size before chunking
        max_chunked_sequence_length = int(math.ceil(hidden_states.shape[1] / get_sequence_parallel_world_size())) * get_sequence_parallel_world_size()
        sequence_pad_amount = max_chunked_sequence_length - hidden_states.shape[1]
        hidden_states = torch.cat(
            [
                hidden_states,
                torch.zeros(
                    batch_size,
                    sequence_pad_amount,
                    hidden_states.shape[2],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
            ],
            dim=1,
        )
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[
            get_sequence_parallel_rank()
        ]

        if ts_seq_len is not None:
            temb = torch.cat(
                [
                    temb,
                    torch.zeros(
                        batch_size,
                        sequence_pad_amount,
                        temb.shape[2],
                        device=temb.device,
                        dtype=temb.dtype,
                    ),
                ],
                dim=1,
            )
            timestep_proj = torch.cat(
                [
                    timestep_proj,
                    torch.zeros(
                        batch_size,
                        sequence_pad_amount,
                        timestep_proj.shape[2],
                        timestep_proj.shape[3],
                        device=timestep_proj.device,
                        dtype=timestep_proj.dtype,
                    ),
                ],
                dim=1,
            )
            temb = torch.chunk(temb, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            timestep_proj = torch.chunk(timestep_proj, get_sequence_parallel_world_size(), dim=1)[
                get_sequence_parallel_rank()
            ]

        freqs_cos, freqs_sin = rotary_emb

        def get_rotary_emb_chunk(freqs, sequence_pad_amount):
            freqs = torch.cat(
                [
                    freqs,
                    torch.zeros(
                        1, sequence_pad_amount, freqs.shape[2], freqs.shape[3], device=freqs.device, dtype=freqs.dtype
                    ),
                ],
                dim=1,
            )
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos, sequence_pad_amount)
        freqs_sin = get_rotary_emb_chunk(freqs_sin, sequence_pad_amount)
        rotary_emb = (freqs_cos, freqs_sin)

        # Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)

        # Remove padding and reshape
        hidden_states = hidden_states[:, : math.prod([post_patch_num_frames, post_patch_height, post_patch_width]), :]
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    new_forward_1 = new_forward.__get__(transformer)
    transformer.forward = new_forward_1
    for block in transformer.blocks:
        block.attn1.processor = xFuserWanAttnProcessor()
        block.attn2.processor = xFuserWanAttnProcessor()

    if transformer_2 is not None:
        new_forward_2 = new_forward.__get__(transformer_2)
        transformer_2.forward = new_forward_2
        for block in transformer_2.blocks:
            block.attn1.processor = xFuserWanAttnProcessor()
            block.attn2.processor = xFuserWanAttnProcessor()


def _parallelize_hunyuan_video_transformer(pipe: Any) -> None:
    """Patch HunyuanVideo transformer for USP+SP parallel with xFuser attention processor."""
    transformer = pipe.transformer

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logging.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        assert batch_size % get_classifier_free_guidance_world_size() == 0, (
            f"Cannot split dim 0 of hidden_states ({batch_size}) into "
            f"{get_classifier_free_guidance_world_size()} parts."
        )

        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb, _ = self.time_text_embed(timestep=timestep, pooled_projection=pooled_projections, guidance=guidance)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1)
        hidden_states = hidden_states.flatten(1, 3)

        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(), dim=0)[
            get_classifier_free_guidance_rank()
        ]
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[
            get_sequence_parallel_rank()
        ]

        encoder_attention_mask = encoder_attention_mask.to(torch.bool).any(dim=0)
        encoder_hidden_states = encoder_hidden_states[:, encoder_attention_mask, :]
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(), dim=0)[
            get_classifier_free_guidance_rank()
        ]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2
            )[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = image_rotary_emb

        def get_rotary_emb_chunk(freqs):
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=0)[get_sequence_parallel_rank()]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        image_rotary_emb = (freqs_cos, freqs_sin)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    None,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    None,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, temb, None, image_rotary_emb)

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, temb, None, image_rotary_emb)

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)
        hidden_states = get_cfg_group().all_gather(hidden_states, dim=0)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )

        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    # Set xFuser attention processor
    from xfuser.model_executor.layers.attention_processor import xFuserHunyuanVideoAttnProcessor2_0
    assert xFuserHunyuanVideoAttnProcessor2_0 is not None
    for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
        block.attn.processor = xFuserHunyuanVideoAttnProcessor2_0()


def ensure_model_downloaded(model_id: str, cache_dir: str | None = None) -> str:
    try:
        logger.info(f"Downloading model {model_id}...")
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
        )
        logger.info(f"Model {model_id} is available at: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        raise


def setup_pipeline(
    model_id: str,
    engine_config: Any,
    local_rank: int,
    args: DiffusionArgs
) -> tuple[Any, int]:
    """Setup the appropriate pipeline based on model_id."""
    if model_id not in PIPELINE_CONFIGS:
        raise ValueError(f"Unsupported model_id: {model_id}")
    
    config = PIPELINE_CONFIGS[model_id]
    model_type = get_model_type_from_id(model_id)

    hf_cache_dir = os.environ.get("HF_HOME", None)
    local_model_dir = None
    if local_rank == 0:
        local_model_dir = ensure_model_downloaded(model_id, hf_cache_dir)
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
    
    # For ConsisID, use a local directory path for both pipeline and face model utils
    if get_model_type_from_id(model_id) == "ConsisID":
        if local_model_dir is None:
            local_model_dir = ensure_model_downloaded(model_id, hf_cache_dir)
        engine_config.model_config.model = local_model_dir
        model_id = local_model_dir

    # For ConsisID, use a local directory path for both pipeline and face model utils
    if get_model_type_from_id(model_id) == "ConsisID" and local_model_dir is not None:
        if local_model_dir is None:
            local_model_dir = ensure_model_downloaded(model_id, hf_cache_dir)
        engine_config.model_config.model = local_model_dir
        model_id = local_model_dir

    cache_args = {
        "use_teacache": False,
        "use_fbcache": False,
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": args.workload.inference_steps,
    }
    
    # Handle T5 encoder if needed
    text_encoder_kwargs = {}
    if config["needs_t5"]:
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder=config["t5_subfolder"],
            torch_dtype=config["dtype"],
        )

        text_encoder_kwargs[config["t5_subfolder"]] = text_encoder
    
    # if args.use_fp8_t5_encoder:
    #     try:
    #         from optimum.quanto import freeze, qfloat8, quantize
    #         logging.info(f"rank {local_rank} quantizing text encoder 2")
    #         quantize(text_encoder, weights=qfloat8)
    #         freeze(text_encoder)
    #     except ImportError:
    #         logging.warning("optimum.quanto not available, skipping T5 quantization")

    # Initialize pipeline
    pipeline_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "engine_config": engine_config,
        "torch_dtype": config["dtype"],
        **text_encoder_kwargs
    }
    
    # Add cache args for models that support it
    if get_model_type_from_id(model_id) == "Flux":
        pipeline_kwargs["cache_args"] = cache_args

    # WAN: use diffusers pipeline with xFuser runtime integration
    if get_model_type_from_id(model_id) == "Wan":
        pipe = WanPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=config["dtype"],
        )
        pipe.scheduler.config.flow_shift = 12
        initialize_runtime_state(pipe, engine_config)
        _parallelize_wan_transformer(pipe)
        pipe = pipe.to(f"cuda:{local_rank}")
    # HunyuanVideo: use diffusers pipeline with xFuser runtime integration
    elif get_model_type_from_id(model_id) == "HunyuanVideo":
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path=model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            revision="refs/pr/18",
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            transformer=transformer,
            torch_dtype=config["dtype"],  # float16
            revision="refs/pr/18",
        )
        initialize_runtime_state(pipe, engine_config)
        get_runtime_state().set_video_input_parameters(
            height=args.workload.height,
            width=args.workload.width,
            num_frames=getattr(args.workload, "num_frames"),
            batch_size=args.workload.batch_size,
            num_inference_steps=args.workload.inference_steps,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )
        _parallelize_hunyuan_video_transformer(pipe)
        pipe = pipe.to(f"cuda:{local_rank}")
    else:
        pipe = config["pipeline_class"].from_pretrained(**pipeline_kwargs)
        pipe = pipe.to(f"cuda:{local_rank}")

    # Attach VAE temporal decoder for Latte video pipeline
    if model_type == "Latte":
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            model_id,
            subfolder="vae_temporal_decoder",
            torch_dtype=config["dtype"],
        ).to(f"cuda:{local_rank}")
        pipe.vae = vae

    return pipe


def get_inference_kwargs(
    request: DiffusionRequest,
    args: DiffusionArgs,
) -> Any:
    # will use the default max_sequence_length and guidance_scale for each model
    model_type = get_model_type_from_id(args.workload.model_id)
    inference_kwargs = {
        "prompt": request.prompts,
        "height": args.workload.height,
        "width": args.workload.width,
        "num_inference_steps": args.workload.inference_steps,
        "output_type": "pil" if args.save_images else "latent",
        "generator": torch.Generator(device="cuda").manual_seed(args.workload.seed),
    }
    
    # Add video parameters
    if hasattr(args.workload, 'num_frames'):
        if model_type == "Latte":
            inference_kwargs["video_length"] = args.workload.num_frames
        else:
            inference_kwargs["num_frames"] = args.workload.num_frames
    # if hasattr(args.workload, 'fps'):
    #     inference_kwargs["fps"] = args.workload.fps
    
    return inference_kwargs


def save_generated_media(
    pipe: Any,
    output: Any,
    request: Any,
    args: DiffusionArgs,
    output_dir: Path,
    iteration_idx: int = 0
):
    """Save images or videos depending on pipeline output.

    - Image case: output.images (list[PIL.Image])
    - Video case: output.frames (Tensor[B, T, C, H, W])
    """
    if not args.save_images:
        return

    should_save = pipe.is_dp_last_group() if hasattr(pipe, 'is_dp_last_group') else is_dp_last_group()
    if not should_save:
        return
    
    dp_group_index = get_data_parallel_rank()
    num_dp_groups = get_data_parallel_world_size()
    num_prompts = len(request.prompts)
    dp_batch_size = (num_prompts + num_dp_groups - 1) // num_dp_groups

    # Save images
    if hasattr(output, "images") and output.images is not None:
        for i, image in enumerate(output.images):
            image_rank = dp_group_index * dp_batch_size + i
            if image_rank < num_prompts:
                prompt_text = request.prompts[image_rank]
                # Extract a clean name from the prompt
                words = prompt_text.split()[:5]  # Take first 5 words
                safe_name = "_".join(word for word in words if word.isalnum() or word in ['-', '_'])[:30]
                safe_name = safe_name.replace(' ', '_').lower()
                filename = f"i{iteration_idx}-{image_rank}_{safe_name}.png"
                image_path = output_dir / filename
                image.save(image_path)
                logger.info(f"Saved image {image_rank} of {iteration_idx} to {image_path}")
        return

    # Save videos
    if hasattr(output, "frames") and output.frames is not None:
        fps = getattr(args.workload, "fps", 8)
        for i, video_frames in enumerate(output.frames):
            video_rank = dp_group_index * dp_batch_size + i
            if video_rank < num_prompts:
                prompt_text = request.prompts[video_rank]
                # Extract a clean name from the prompt
                words = prompt_text.split()[:5]  # Take first 5 words
                safe_name = "_".join(word for word in words if word.isalnum() or word in ['-', '_'])[:30]
                safe_name = safe_name.replace(' ', '_').lower()
                filename = f"i{iteration_idx}-{video_rank}_{safe_name}.mp4"
                video_path = output_dir / filename
                export_to_video(video_frames, video_path, fps=fps)
                logger.info(f"Saved video {video_rank} of {iteration_idx} to {video_path}")


def save_results(
    args: DiffusionArgs,
    benchmark_duration: float,
    total_energy_result: Any = None,
    iter_energy_results: list[Any] = None,
    local_rank: int = 0,
    power_timeline: Any | None = None,
    temperature_timeline: Any | None = None,
    benchmark_start_time: float | None = None,
    benchmark_end_time: float | None = None,
) -> None:
    if local_rank != 0:
        return
        
    # Calculate metrics
    num_images = args.workload.batch_size * args.benchmark_iters
    throughput = num_images / benchmark_duration if benchmark_duration > 0 else 0.0
    avg_time_per_image = benchmark_duration / num_images if num_images > 0 else 0.0
    
    # Prepare results dictionary
    result_json: dict[str, Any] = {}
    
    # Setup information
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["model_id"] = args.workload.model_id
    result_json["batch_size"] = args.workload.batch_size
    result_json["num_iterations"] = args.benchmark_iters
    
    # Generation parameters
    result_json["height"] = args.workload.height
    result_json["width"] = args.workload.width 
    result_json["inference_steps"] = args.workload.inference_steps
    result_json["seed"] = args.workload.seed
    
    # Performance metrics  
    result_json["total_images"] = num_images
    result_json["total_time"] = benchmark_duration
    result_json["throughput_images_per_sec"] = throughput
    result_json["avg_time_per_image"] = avg_time_per_image
    
    # Energy metrics
    if total_energy_result:
        result_json["total_energy"] = total_energy_result.total_energy
        result_json["energy_per_image"] = total_energy_result.total_energy / num_images if num_images > 0 else 0.0
        result_json["energy_measurement"] = asdict(total_energy_result)
    
    for i, iter_energy_result in enumerate(iter_energy_results):
        result_json[f"iter{i}_energy_measurement"] = asdict(iter_energy_result)
    
    # Configuration details
    result_json["configurations"] = {
        "ulysses_degree": args.workload.ulysses_degree,
        "ring_degree": args.workload.ring_degree,
        "use_torch_compile": args.workload.use_torch_compile,
    }

    # Timeline details (power and temperature over time)
    if benchmark_start_time is not None and benchmark_end_time is not None:
        result_json["timeline"] = {
            "benchmark_start_time": benchmark_start_time,
            "benchmark_end_time": benchmark_end_time,
            "power": power_timeline,
            "temperature": temperature_timeline,
        }

    # Save results
    result_file = args.workload.to_path(of="results")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)
    
    logger.info(f"Results saved to {result_file}")
    
    # Log summary
    logger.info("[Diffusion Benchmark Results]")
    logger.info("%-40s: %s", "Model ID", args.workload.model_id)
    logger.info("%-40s: %d", "Batch Size", args.workload.batch_size)
    logger.info("%-40s: %d", "Total Images Requested", num_images)
    logger.info("%-40s: %.2f", "Total Time (s)", benchmark_duration)
    logger.info("%-40s: %.2f", "Images per Second", throughput)
    logger.info("%-40s: %.2f", "Seconds per Image", avg_time_per_image)
    
    if total_energy_result:
        logger.info("%-40s: %.2f", "Total Energy (J)", total_energy_result.total_energy)
        logger.info("%-40s: %.2f", "Energy per Image (J)", total_energy_result.total_energy / num_images)


def main(args: DiffusionArgs) -> None:
    """Main benchmark function."""
    logger.info("%s", args)
    assert isinstance(args.workload, DiffusionWorkloadConfig)

    result_file = args.workload.to_path(of="results")
    if result_file.exists() and not args.overwrite_results:
        logger.info(
            "Result file %s already exists. Exiting immediately. "
            "Specify --overwrite_results to run the benchmark and overwrite results.",
            result_file,
        )
        return

    # # fix lamdalab segfault error
    # os.environ["NCCL_NET"] = "Socket"
    # os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    zeus_monitor = None
    power_monitor = None
    temperature_monitor = None
    if os.environ.get("LOCAL_RANK", "0") == "0":
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            show_env()
        logger.info("Zeus environment information:\n%s", buffer.getvalue())

        zeus_monitor = ZeusMonitor()
        power_monitor = PowerMonitor(update_period=0.1)
        temperature_monitor = TemperatureMonitor(update_period=0.5)
    
    random.seed(args.workload.seed)
    np.random.seed(args.workload.seed)
    torch.manual_seed(args.workload.seed)

    requests = args.workload.load_requests(args.warmup_iters, args.benchmark_iters)

    # Setup xFuser arguments
    logger.info(f"Setting up xFuser args")
    xfuser_args = xFuserArgs(
        model=args.workload.model_id,
        ulysses_degree=args.workload.ulysses_degree,
        ring_degree=args.workload.ring_degree,
        height=args.workload.height,
        width=args.workload.width,
        num_inference_steps=args.workload.inference_steps,
        seed=args.workload.seed,
        prompt=requests[0].prompts,
        use_torch_compile=args.workload.use_torch_compile,
    )
    engine_config, input_config = xfuser_args.create_config()

    world_group = get_world_group()
    local_rank = world_group.local_rank
    world_size = world_group.world_size

    # Setup xFuser pipeline
    logger.info(f"Setting up xFuser pipeline")
    pipe = setup_pipeline(args.workload.model_id, engine_config, local_rank, args)

    # Enable VAE tiling and slicing for text-to-video workloads
    if isinstance(args.workload, TextToVideo):
        pipe.vae.enable_tiling()
        logger.info("Enabled VAE tiling for text-to-video pipeline.")
        pipe.vae.enable_slicing()
        logger.info("Enabled VAE slicing for text-to-video pipeline.")

    # Prepare/warmup pipeline
    model_type = get_model_type_from_id(args.workload.model_id)
    dtype = PIPELINE_CONFIGS[args.workload.model_id]["dtype"]
    if model_type == "CogVideo" or model_type == "Latte" or model_type == "Wan" or model_type == "HunyuanVideo":
        logger.info(f"Warming up pipeline with 1-step inference")
        warmup_kwargs = get_inference_kwargs(requests[0], args)
        warmup_kwargs["num_inference_steps"] = 1  # Single step warmup
        if model_type == "Latte":
            # Ensure inputs and weights run under fp16 autocast to avoid dtype mismatches
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = pipe(**warmup_kwargs)
        else:
            _ = pipe(**warmup_kwargs)
    elif model_type == "ConsisID":
        logger.info("ConsisID detected: skipping prepare_run; will warm up via forward call.")
    else:
        logger.info(f"Preparing xFuser pipeline with prepare_run")
        pipe.prepare_run(input_config, steps=args.workload.inference_steps)

    # Warmup iterations with different requests
    logger.info(f"Running {args.warmup_iters} warmup iterations with different requests")
    output_kind = "video_outputs" if getattr(args.workload, "num_frames", None) else "image_outputs"
    output_dir = args.workload.to_path(of=output_kind)
    
    # Prepare face models once for ConsisID
    consisid_face_models = None
    consisid_face_inputs = None
    if model_type == "ConsisID":
        device = torch.device(f"cuda:{local_rank}")
        model_root = getattr(engine_config.model_config, "model", args.workload.model_id)
        consisid_face_models = prepare_face_models(model_root, device=device, dtype=dtype)
        # Precompute face embeddings once and reuse across all iterations
        (
            face_helper_1,
            face_helper_2,
            face_clip_model,
            face_main_model,
            eva_transform_mean,
            eva_transform_std,
        ) = consisid_face_models
        img_file_path = getattr(args.workload, "img_file_path", None)
        consisid_face_inputs = process_face_embeddings_infer(
            face_helper_1,
            face_clip_model,
            face_helper_2,
            eva_transform_mean,
            eva_transform_std,
            face_main_model,
            device,
            dtype,
            img_file_path,
            is_align_face=True,
        )
    
    for i in range(args.warmup_iters):
        logger.info(f"Warmup iteration {i+1}/{args.warmup_iters}")
        warmup_request = requests[i]
        if model_type == "ConsisID":
            # Reuse precomputed ConsisID-specific inputs
            (id_cond, id_vit_hidden, image, face_kps) = consisid_face_inputs
            warmup_kwargs = get_inference_kwargs(warmup_request, args)
            warmup_kwargs.update(
                {
                    "image": image,
                    "id_vit_hidden": id_vit_hidden,
                    "id_cond": id_cond,
                    "kps_cond": face_kps,
                    "use_dynamic_cfg": False,
                }
            )
            warmup_output = pipe(**warmup_kwargs)
        else:
            warmup_kwargs = get_inference_kwargs(warmup_request, args)
            if model_type == "Latte":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    warmup_output = pipe(**warmup_kwargs)
            else:
                warmup_output = pipe(**warmup_kwargs)

    # Benchmark iterations with different requests
    logger.info(f"Start running {args.benchmark_iters} benchmark iterations")
    iter_energy_results = []
    torch.cuda.synchronize()
    torch.distributed.barrier(device_ids=[local_rank])
    benchmark_start_time = time.time()
    
    if zeus_monitor:
        zeus_monitor.begin_window("entire_benchmark")
    
    for i in range(args.benchmark_iters):
        request_idx = args.warmup_iters + i
        benchmark_request = requests[request_idx]
        if model_type == "ConsisID":
            # Reuse precomputed ConsisID-specific inputs
            (id_cond, id_vit_hidden, image, face_kps) = consisid_face_inputs
            benchmark_kwargs = get_inference_kwargs(benchmark_request, args)
            benchmark_kwargs.update(
                {
                    "image": image,
                    "id_vit_hidden": id_vit_hidden,
                    "id_cond": id_cond,
                    "kps_cond": face_kps,
                    "use_dynamic_cfg": False,
                }
            )
        else:
            benchmark_kwargs = get_inference_kwargs(benchmark_request, args)
        
        logger.info(f"Benchmark iteration {i+1}/{args.benchmark_iters}")
        if zeus_monitor:
            zeus_monitor.begin_window("iteration")
        if model_type == "Latte":
            with torch.autocast(device_type="cuda", dtype=dtype):
                benchmark_output = pipe(**benchmark_kwargs)
        else:
            benchmark_output = pipe(**benchmark_kwargs)
        if zeus_monitor:
            iter_energy_results.append(zeus_monitor.end_window("iteration"))

        save_generated_media(pipe, benchmark_output, benchmark_request, args, output_dir, i)

    torch.cuda.synchronize()
    torch.distributed.barrier(device_ids=[local_rank])

    total_energy_result = None
    if zeus_monitor:
        total_energy_result = zeus_monitor.end_window("entire_benchmark")
    benchmark_end_time = time.time()

    power_timeline = None
    temperature_timeline = None
    if power_monitor and temperature_monitor:
        power_timeline = power_monitor.get_all_power_timelines(
            start_time=benchmark_start_time,
            end_time=benchmark_end_time,
        )
        temperature_timeline = temperature_monitor.get_temperature_timeline(
            start_time=benchmark_start_time,
            end_time=benchmark_end_time,
        )
    
    benchmark_duration = benchmark_end_time - benchmark_start_time
    logger.info(f"End running diffusion benchmark, duration: {benchmark_duration:.2f}s")

    if local_rank == 0:
        save_results(
            args,
            benchmark_duration,
            total_energy_result,
            iter_energy_results,
            local_rank,
            power_timeline,
            temperature_timeline,
            benchmark_start_time,
            benchmark_end_time,
        )
    
    get_runtime_state().destroy_distributed_env()


# TODO: handle server log
if __name__ == "__main__":
    args = tyro.cli(DiffusionArgs[TextToImage | TextToVideo])

    # Set up logging
    # Only rank 0 should write to the driver log file to avoid conflicts
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # Create handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s: %(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)
    
    handlers = [stream_handler]
    
    if local_rank == 0:
        file_handler = logging.FileHandler(args.workload.to_path(of="driver_log"), mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger with force=True to override any existing configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s: %(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    try:
        main(args)
    except Exception as e:
        logger.exception("An error occurred during the benchmark: %s", e)
        raise 