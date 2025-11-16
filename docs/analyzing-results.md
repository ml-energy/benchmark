# Analyzing Results

## Directory Structure

Results are saved to `{base_dir}` (typically `run/`). The hierarchical structure is:

```
run/
├── llm/                                         # Text-only tasks
│   └── {task}/                                  # e.g., gpqa, lm-arena-chat, sourcegraph-fim
│       ├── requests/                            # Model-independent request data (shared)
│       │   └── {dataset_params}/                # e.g., num_requests+100+seed+48105
│       │       ├── requests.json                # Prompts, completions, multimodal content
│       │       └── multimodal_dump/             # [Optional] Dumped images/videos/audio
│       ├── tokenization/                        # Model-dependent tokenization data
│       │   └── {model_id}/                      # e.g., Qwen/Qwen3-8B
│       │       └── {dataset_params}.json        # Token counts and token IDs
│       └── results/                             # Benchmark results
│           └── {model_id}/
│               └── {gpu_model}/                 # e.g., H100, B200
│                   └── {runtime_params}/        # e.g., num_gpus+1+max_num_seqs+128+...
│                       ├── results.json         # Benchmark metrics and outputs
│                       ├── prometheus.json      # Time-series metrics from vLLM
│                       ├── driver.log           # Driver execution log
│                       ├── server.log           # vLLM server log
│                       ├── requests.json@       # Symlink to shared requests
│                       └── tokenization.json@   # Symlink to shared tokenization
│
└── mllm/                             # Multimodal tasks
    └── {task}/                       # e.g., image-chat, video-chat, audio-chat
        └── ...                       # Same structure as llm/
```

### Design Principles

1. **Data Deduplication**: `requests.json` is shared across all models and runtime configurations for the same task and dataset parameters, avoiding duplication of potentially large files (especially for video data).

2. **Tokenization Sharing**: Tokenization data is shared across different runtime configurations (e.g., different `max_num_seqs`) but regenerated when the model changes, since tokenization is model-dependent.

3. **Symlinks for Context**: Results directories contain symlinks to the corresponding `requests.json` and `tokenization.json` files, making it easy to understand which data was used without duplicating files.

4. **Hierarchical Organization**:
   - **Task-level**: Shared data that depends only on the task and dataset parameters
   - **Model-level**: Tokenization that depends on both task and model
   - **Configuration-level**: Results that depend on task, model, GPU, and runtime parameters

## Output Files

### `results.json`

Main benchmark results file containing metrics and per-request data.

**Top-level structure:**
```json
{
  "model_id": "Qwen/Qwen3-8B",
  "gpu_model": "H100",
  "num_gpus": 1,
  "max_num_seqs": 128,
  "num_prompts": 1024,
  "completed": 1024,
  "duration": 135.2,
  "steady_state_duration": 97.6,
  "steady_state_energy": 54631.48,
  "steady_state_energy_per_token": 0.093545,
  "output_throughput": 4890.37,
  "total_output_tokens": 661234,
  "request_rate": null,
  "burstiness": null,
  "max_concurrency": null,
  "results": [ /* per-request results */ ],
  "steady_state_measurement": { /* GPU energy/temp */ }
}
```

**Key metrics:**
- `duration`: Total benchmark duration in seconds
- `steady_state_duration`: Duration of steady-state measurement window (excludes warmup/cooldown)
- `steady_state_energy`: Total GPU energy consumption during steady state (Joules)
- `steady_state_energy_per_token`: Energy efficiency metric (J/token)
- `output_throughput`: Output tokens per second
- `completed` vs `num_prompts`: Check these match to verify all requests completed

**Per-request results** (`results` array):
```json
{
  "prompt": "User prompt text...",
  "output_text": "Model response...",
  "reasoning_output_text": "Reasoning tokens (if applicable)",
  "input_len": 245,
  "output_len": 646,
  "dataset_output_len": 650,
  "ttft": 0.156,
  "itl": [0.012, 0.011, 0.013, ...],
  "latency": 8.234,
  "energy": 52.1,
  "success": true,
  "error": ""
}
```

**Per-request fields:**
- `ttft`: Time to first token (seconds)
- `itl`: Inter-token latencies (seconds per token)
- `latency`: Total request latency (seconds)
- `energy`: Energy consumed for this request (Joules)
- `success`: Whether request completed successfully
- `error`: Error message if `success` is false

### `prometheus.json`

Time-series metrics collected from vLLM's Prometheus endpoint at 1-second intervals. Structure:

```json
{
  "collection_interval": 1.0,
  "steady_state_start_time": 1763283464.529182,
  "steady_state_end_time": 1763283562.135392,
  "timeline": [
    {
      "timestamp": 1763283463.085004,
      "metrics": "# vLLM Prometheus metrics in text format..."
    }
  ],
  "steady_state_stats": {
    "vllm:kv_cache_usage_perc": 0.324,
    "vllm:num_requests_running": {...}
  }
}
```

**Timeline entries** contain full Prometheus metrics snapshots including:
- `vllm:kv_cache_usage_perc`: KV cache utilization (0-1 range)
- `vllm:num_requests_running`: Active requests
- `vllm:num_requests_waiting`: Queued requests
- Request latency histograms
- Token count histograms

**Note:** The number of timeline entries should roughly equal `duration` in seconds. Some multimodal workloads may have slightly lower collection rates (0.7-0.9 ratio) due to processing overhead.

## Analyzing Results

### Automated Validation (Recommended)

Validate all benchmark results against expectations:

```bash
# Basic validation
python scripts/validate_results.py

# Show details for all runs
python scripts/validate_results.py --verbose

# Validate specific directory
python scripts/validate_results.py --base-dir run/llm/lm-arena-chat/
```

The script validates 7 expectations:

1. **Files Present**: All result files exist
2. **Completion**: All requests finished (`completed == num_prompts`)
3. **Request Success**: All requests succeeded
4. **Steady State Duration**: At least 30 seconds long
5. **Prometheus Collection**: ~1 collection/second for both total and steady-state (>=75%)
6. **Valid Metrics**: Throughput, energy > 0, under broadly-defined reasonable ranges
7. **No Crashes**: No `RuntimeError`, CUDA assertions, or `EngineDeadError` in logs

### Quick Manual Checks

```bash
# Check completion status
jq '{completed: .completed, num_prompts: .num_prompts, throughput: .output_throughput}' results.json

# Check for failed requests
jq '[.results[] | select(.success == false)] | length' results.json

# Check steady-state ratio (should be >50% for most workloads)
jq '{duration: .duration, steady: .steady_state_duration, ratio: (.steady_state_duration / .duration)}' results.json
```
