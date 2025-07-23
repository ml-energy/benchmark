#!/bin/bash
set -x

export PYTHONPATH=/workspaces/xDiT:$PWD:$PYTHONPATH

mkdir -p ./results

# Dataset configuration
BATCH_SIZE=4
DATASET_SEED=42
PROMPT_CACHE_DIR="./prompt_cache"

# Task args (same as original)
TASK_ARGS="--height 512 --width 512 --no_use_resolution_binning --guidance_scale 3.5"

# Dataset args
DATASET_ARGS="--use_dataset --batch_size $BATCH_SIZE --dataset_seed $DATASET_SEED --prompt_cache_dir $PROMPT_CACHE_DIR"

# Cache args (uncomment to use)
# CACHE_ARGS="--use_teacache"
# CACHE_ARGS="--use_fbcache"

# Parallel configuration (same as original)
ULYSS_DEGREE=2
RING_DEGREE=2
N_GPUS=$((ULYSS_DEGREE*RING_DEGREE))
PARALLEL_ARGS="--ulysses_degree $ULYSS_DEGREE --ring_degree $RING_DEGREE"

# Data parallel configuration (for batch processing)
# DATA_PARALLEL_ARGS="--data_parallel_degree 2"

# CFG parallel (uncomment to use)
# CFG_ARGS="--use_cfg_parallel"

# PipeFusion args (uncomment to use)
# PIPEFUSION_ARGS="--pipefusion_parallel_degree 2 --num_pipeline_patch 8"

# For high-resolution images, use latent output to avoid VAE (for speed testing)
# OUTPUT_ARGS="--output_type latent"

# Parallel VAE (uncomment to use)
# PARALLLEL_VAE="--use_parallel_vae"

# Compile options (uncomment to use)
# COMPILE_FLAG="--use_torch_compile"

# Quantization (uncomment to use)
# QUANTIZE_FLAG="--use_fp8_t5_encoder"

# Memory optimization (uncomment to use)
# MEMORY_ARGS="--enable_sequential_cpu_offload"

# Set visible GPUs (uncomment and modify as needed)
# export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Running Flux batch inference with dataset:"
echo "  Batch size: $BATCH_SIZE"
echo "  Random seed: $DATASET_SEED"
echo "  Cache directory: $PROMPT_CACHE_DIR"
echo "  GPUs: $N_GPUS"

torchrun --nproc_per_node=$N_GPUS flux_example.py \
--model black-forest-labs/FLUX.1-dev \
$PARALLEL_ARGS \
$DATA_PARALLEL_ARGS \
$TASK_ARGS \
$DATASET_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps 28 \
--warmup_steps 1 \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
$CACHE_ARGS \
$MEMORY_ARGS

echo ""
echo "Alternative usage with manual prompts (without dataset):"
echo "torchrun --nproc_per_node=$N_GPUS flux_example.py \\"
echo "  --model /cfs/dit/FLUX.1-dev/ \\"
echo "  --prompt \"first prompt\" --prompt \"second prompt\" \\"
echo "  $PARALLEL_ARGS $TASK_ARGS --num_inference_steps 28" 