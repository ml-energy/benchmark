#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id>"
    echo "Where gpu_id is 0-7"
    exit 1
fi

GPU_ID=$1

# Validate GPU ID
if [ "$GPU_ID" -lt 0 ] || [ "$GPU_ID" -gt 7 ]; then
    echo "Error: GPU ID must be between 0 and 7"
    exit 1
fi

AUDIO_DATA_DIR="/data/hfcache/hub/datasets--Fhrozen--FSD50k/snapshots/ccf1acaff12f3f4a4c10052dddafbb6d22152c9b/clips/dev"
MODEL_ID="Qwen/Qwen2-Audio-7B"
BASE_DIR="run/mllm/${MODEL_ID}"
SERVER_IMAGE="leaderboard/vllm:v0.10.0rc1"
NUM_REQUESTS=2048

# Array of max_num_seqs values corresponding to GPU IDs 0-7
MAX_NUM_SEQS=(8 16 32 64 128 256 512 1024)

# Get the max_num_seqs value for the specified GPU
MAX_SEQS=${MAX_NUM_SEQS[$GPU_ID]}

echo "Running benchmark on GPU $GPU_ID with max_num_seqs=$MAX_SEQS"

CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python3 -m mlenergy.llm.benchmark \
    --server-image $SERVER_IMAGE \
    workload:audio-chat \
    --workload.model-id $MODEL_ID \
    --workload.base-dir $BASE_DIR \
    --workload.num-requests $NUM_REQUESTS \
    --workload.num-audios 1 \
    --workload.audio-data-dir $AUDIO_DATA_DIR \
    --workload.max-num-seqs $MAX_SEQS
