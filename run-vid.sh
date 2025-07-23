VIDEO_DATA_DIR="/turbo/llava_video_178k"
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
BASE_DIR="run/mllm/${MODEL_ID}"
SERVER_IMAGE="vllm/vllm-openai:v0.9.0"
NUM_REQUESTS=2000

CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -m mlenergy.llm.benchmark --server-image $SERVER_IMAGE workload:video-chat --workload.model-id $MODEL_ID --workload.base-dir $BASE_DIR --workload.num-requests $NUM_REQUESTS --workload.num-videos 1 --workload.video-data-dir $VIDEO_DATA_DIR --workload.max-num-seqs 8 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -m mlenergy.llm.benchmark --server-image $SERVER_IMAGE workload:video-chat --workload.model-id $MODEL_ID --workload.base-dir $BASE_DIR --workload.num-requests $NUM_REQUESTS --workload.num-videos 1 --workload.video-data-dir $VIDEO_DATA_DIR --workload.max-num-seqs 16 &
CUDA_VISIBLE_DEVICES=2 .venv/bin/python3 -m mlenergy.llm.benchmark --server-image $SERVER_IMAGE workload:video-chat --workload.model-id $MODEL_ID --workload.base-dir $BASE_DIR --workload.num-requests $NUM_REQUESTS --workload.num-videos 1 --workload.video-data-dir $VIDEO_DATA_DIR --workload.max-num-seqs 32 &
CUDA_VISIBLE_DEVICES=3 .venv/bin/python3 -m mlenergy.llm.benchmark --server-image $SERVER_IMAGE workload:video-chat --workload.model-id $MODEL_ID --workload.base-dir $BASE_DIR --workload.num-requests $NUM_REQUESTS --workload.num-videos 1 --workload.video-data-dir $VIDEO_DATA_DIR --workload.max-num-seqs 64 &
CUDA_VISIBLE_DEVICES=4 .venv/bin/python3 -m mlenergy.llm.benchmark --server-image $SERVER_IMAGE workload:video-chat --workload.model-id $MODEL_ID --workload.base-dir $BASE_DIR --workload.num-requests $NUM_REQUESTS --workload.num-videos 1 --workload.video-data-dir $VIDEO_DATA_DIR --workload.max-num-seqs 128 &
CUDA_VISIBLE_DEVICES=5 .venv/bin/python3 -m mlenergy.llm.benchmark --server-image $SERVER_IMAGE workload:video-chat --workload.model-id $MODEL_ID --workload.base-dir $BASE_DIR --workload.num-requests $NUM_REQUESTS --workload.num-videos 1 --workload.video-data-dir $VIDEO_DATA_DIR --workload.max-num-seqs 256 &
CUDA_VISIBLE_DEVICES=6 .venv/bin/python3 -m mlenergy.llm.benchmark --server-image $SERVER_IMAGE workload:video-chat --workload.model-id $MODEL_ID --workload.base-dir $BASE_DIR --workload.num-requests $NUM_REQUESTS --workload.num-videos 1 --workload.video-data-dir $VIDEO_DATA_DIR --workload.max-num-seqs 512 &
CUDA_VISIBLE_DEVICES=7 .venv/bin/python3 -m mlenergy.llm.benchmark --server-image $SERVER_IMAGE workload:video-chat --workload.model-id $MODEL_ID --workload.base-dir $BASE_DIR --workload.num-requests $NUM_REQUESTS --workload.num-videos 1 --workload.video-data-dir $VIDEO_DATA_DIR --workload.max-num-seqs 1024 &
