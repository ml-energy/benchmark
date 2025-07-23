# Diffusion Model Benchmark

## Task Benchmark

### Models

#### Text to Image (T2I)

- **Flux**: black-forest-labs/FLUX.1-dev
- **PixArt-Sigma**: 
- **SD3**: 
- **HunyuanDiT**: 

#### Text to Video (T2V)

### Setup

```bash
# Install xDiT (using commit 21dcdcf without Moore Technologies features)
pip install -e "git+https://github.com/xdit-project/xDiT.git@21dcdcf#egg=xfuser[diffusers,flash-attn]"
```

### Basic Usage

#### Text-to-Image (T2I)

```bash
# Single GPU execution
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 mlenergy/diffusion/benchmark.py \
    --ulysses_degree 1 \
    --ring_degree 1 \
    workload:text-to-image \
    --workload.model_id black-forest-labs/FLUX.1-dev \
    --workload.base_dir run/diffusion/text-to-image/FLUX.1-dev \
    --workload.batch_size 4

# Multi-GPU execution
# ULYSSES_DEGREE * RING_DEGREE = 2 * 2 = 4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 mlenergy/diffusion/benchmark.py \
    --ulysses_degree 2 \
    --ring_degree 2 \
    workload:text-to-image \
    --workload.model_id black-forest-labs/FLUX.1-dev \
    --workload.base_dir run/diffusion/text-to-image/FLUX.1-dev \
    --workload.batch_size 4
```

<!-- #### Text-to-Video (T2V) - Coming Soon

```bash
# Run a text-to-video benchmark (when supported)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m mlenergy.diffusion.benchmark \
    workload:t2v \
    --workload.model_id black-forest-labs/FLUX.1-dev \
    --workload.base_dir run/diffusion/flux_t2v \
    --workload.batch_size 5 \
    --workload.height 512 \
    --workload.width 512 \
    --workload.num_frames 16 \
    --workload.fps 8 \
    --ulysses_degree 2 \
    --ring_degree 2
```

### Advanced Configuration

```bash
# High-performance setup with optimizations
python -m mlenergy.diffusion.benchmark \
    workload:t2i \
    --workload.model_id black-forest-labs/FLUX.1-dev \
    --workload.base_dir run/diffusion/flux_optimized \
    --workload.batch_size 100 \
    --workload.height 512 \
    --workload.width 512 \
    --workload.inference_steps 28 \
    --workload.guidance_scale 3.5 \
    --ulysses_degree 2 \
    --ring_degree 2 \
    --use_torch_compile \
    --use_fp8_t5_encoder \
    --use_teacache
```

### Batch Multiple Models

```bash
# Benchmark different models
for model_id in "black-forest-labs/FLUX.1-dev" "stabilityai/stable-diffusion-xl-base-1.0"; do
    python -m mlenergy.diffusion.benchmark \
        workload:t2i \
        --workload.model_id $model_id \
        --workload.base_dir run/diffusion/$(echo $model_id | tr '/' '-') \
        --workload.batch_size 50 \
        --ulysses_degree 2 \
        --ring_degree 2
done
``` -->

<!-- ## Results

Results are saved in the same structured format as LLM benchmarks:

```bash
# Check results structure
tree run

run/
└── diffusion/
    └── flux_t2i/
        └── model-black-forest-labs-FLUX_1-dev+batch-10+size-1024x1024+steps-28+guidance-3.5+seed-42/
            ├── requests.json         # Input prompts and parameters
            ├── results.json         # Detailed metrics and results
            ├── driver_log.txt       # Benchmark execution logs  
            └── image_outputs/       # Generated images (if enabled)
                ├── black-forest-labs-FLUX_1-dev_0001_a_majestic_dragon.png
                └── ...
```

### Results Format

The `results.json` contains comprehensive metrics:

```json
{
  "date": "20241215-143022",
  "model_id": "black-forest-labs/FLUX.1-dev",
  "batch_size": 10,
  "completed": 10,
  "generation_time": 45.2,
  "throughput_images_per_sec": 0.22,
  "avg_time_per_image": 4.52,
  "total_energy_j": 1250.5,
  "energy_per_image_j": 125.05,
  "peak_memory_gb": 18.4,
  "prompts": ["...", "..."],
  "generation_results": [...],
  "configuration": {...},
  "energy_measurement": {...}
}
```

## Configuration Options

### Workload Types

- **workload:t2i** - Text-to-Image generation
- **workload:t2v** - Text-to-Video generation (coming soon)

### Workload Parameters

- `model_id`: Model identifier (e.g. black-forest-labs/FLUX.1-dev)
- `base_dir`: Output directory for results
- `batch_size`: Number of images to generate in one batch
- `height/width`: Output image dimensions
- `inference_steps`: Number of denoising steps
- `guidance_scale`: Classifier-free guidance strength
- `seed`: Random seed for reproducibility

### Model Types

Each model has specific optimized defaults:

| Model | Default Steps | Supports T5 | Default dtype |
|-------|---------------|-------------|---------------|
| FLUX.1-dev | 28 | Yes | bfloat16 |
| SDXL | 30 | No | float16 |
| PixArt-α | 20 | No | float16 |
| PixArt-Σ | 20 | Yes | float16 |
| SD3 | 20 | Yes | float16 |
| HunyuanDiT | 50 | Yes | float16 |

### Performance Options

- `ulysses_degree`: Ulysses attention parallelism degree
- `ring_degree`: Ring attention parallelism degree  
- `use_torch_compile`: Enable PyTorch compilation
- `use_fp8_t5_encoder`: Use FP8 quantization for T5
- `enable_sequential_cpu_offload`: CPU memory offloading
- `use_teacache/use_fbcache`: Attention caching optimizations

## Energy Monitoring

The benchmark integrates with Zeus for GPU energy monitoring:

- **Total Energy**: Complete benchmark energy consumption
- **Energy per Image**: Average energy cost per generated image
- **Peak Memory**: Maximum GPU memory usage
- **Parameter Memory**: Model parameter memory footprint

## Dataset Integration

Prompts are automatically sourced from high-quality datasets:

- **Primary**: `data-is-better-together/open-image-preferences-v1` (cleaned split)
- **Automatic Caching**: Prompts cached locally for consistency
- **Reproducible Sampling**: Seed-based prompt selection

## Comparison with LLM Benchmarks

| Feature | LLM Benchmark | Diffusion Benchmark |
|---------|---------------|-------------------|
| **Metrics** | TTFT, TPOT, throughput | Generation time, images/sec, energy/image |
| **Workloads** | Text generation | Image generation |
| **Energy** | Request-level | Batch-level |
| **Output** | Token sequences | PIL images + metrics |
| **Parallelism** | Request concurrency | Attention parallelism |

Both benchmarks share:
- Zeus energy monitoring
- Structured result format
- Workload configuration system
- Reproducible experiment setup
- Comprehensive logging  -->