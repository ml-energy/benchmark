import logging
import time
import torch
import torch.distributed
import random
import json
import os
from datasets import load_dataset
from transformers.models.t5 import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_tensor_model_parallel_world_size,
    get_data_parallel_world_size,
)
from xfuser.model_executor.cache.diffusers_adapters import apply_cache_on_transformer

from zeus.monitor import ZeusMonitor


def load_prompts_from_dataset(batch_size, seed=42, cache_dir="./prompt_cache"):
    """Load prompts from the open-image-preferences dataset with caching"""
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename based on batch_size and seed
    cache_filename = f"prompts_batch{batch_size}_seed{seed}.json"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Try to load from cache first - look for any cache file with the same seed
    cache_pattern = f"prompts_batch*_seed{seed}.json"
    import glob
    cache_files = glob.glob(os.path.join(cache_dir, cache_pattern))
    
    if cache_files:
        # Use the first cache file found (they all have the same seed)
        cache_file = cache_files[0]
        print(f"Loading prompts from cache: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            all_cached_prompts = cached_data['prompts']
            
            # Take only the first batch_size prompts
            if batch_size <= len(all_cached_prompts):
                selected_prompts = all_cached_prompts[:batch_size]
                print(f"Using first {len(selected_prompts)} prompts from {len(all_cached_prompts)} cached prompts")
                for i, prompt in enumerate(selected_prompts):
                    print(f"  {i+1}: {prompt[:100]}..." if len(prompt) > 100 else f"  {i+1}: {prompt}")
                return selected_prompts
            else:
                print(f"Requested batch_size ({batch_size}) is larger than cached prompts ({len(all_cached_prompts)}), falling back to dataset loading")
        except Exception as e:
            print(f"Error loading cache file: {e}, falling back to dataset loading")
    
    # Load from dataset if cache doesn't exist or failed to load
    print("Loading dataset...")
    ds = load_dataset("data-is-better-together/open-image-preferences-v1")
    
    # Set random seed for reproducible sampling
    random.seed(seed)
    
    # Use the 'cleaned' split which contains 8667 rows
    dataset = ds["cleaned"]
    print("Using 'cleaned' split")
    
    # Extract prompts - the dataset has 'prompt' field
    all_prompts = []
    for item in dataset:
        if isinstance(item, dict) and 'prompt' in item:
            prompt = item['prompt']
            if prompt is not None and prompt.strip():
                all_prompts.append(str(prompt).strip())
    
    # Randomly sample prompts based on batch size
    if batch_size > len(all_prompts):
        print(f"Warning: Requested batch size ({batch_size}) is larger than available prompts ({len(all_prompts)})")
        batch_size = len(all_prompts)
    
    selected_prompts = random.sample(all_prompts, batch_size)
    print(f"Selected {len(selected_prompts)} prompts from dataset")
    
    # Save to cache for next time
    cache_data = {
        'batch_size': batch_size,
        'seed': seed,
        'prompts': selected_prompts,
        'total_available': len(all_prompts)
    }
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    print(f"Prompts saved to cache: {cache_path}")
    
    for i, prompt in enumerate(selected_prompts):
        print(f"  {i+1}: {prompt[:100]}..." if len(prompt) > 100 else f"  {i+1}: {prompt}")
    
    return selected_prompts

def main():
    parser = FlexibleArgumentParser(description="xFuser Flux Batch Dataset Arguments")
    
    # Add dataset-specific arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Number of prompts to sample from dataset")
    parser.add_argument("--dataset_seed", type=int, default=42,
                       help="Random seed for dataset sampling")
    parser.add_argument("--use_dataset", action="store_true",
                       help="Use dataset prompts instead of --prompt argument")
    parser.add_argument("--prompt_cache_dir", type=str, default="./prompt_cache",
                       help="Directory to store cached prompts")
    
    args = xFuserArgs.add_cli_args(parser).parse_args()
    
    # Load prompts from dataset if requested
    if args.use_dataset:
        prompts = load_prompts_from_dataset(
            batch_size=args.batch_size,
            seed=args.dataset_seed,
            cache_dir=args.prompt_cache_dir
        )
        # Override the prompt argument
        args.prompt = prompts
    
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    world_size = get_world_group().world_size
    
    text_encoder_2 = T5EncoderModel.from_pretrained(
        engine_config.model_config.model, 
        subfolder="text_encoder_2", 
        torch_dtype=torch.bfloat16
    )

    if args.use_fp8_t5_encoder:
        try:
            from optimum.quanto import freeze, qfloat8, quantize
            logging.info(f"rank {local_rank} quantizing text encoder 2")
            quantize(text_encoder_2, weights=qfloat8)
            freeze(text_encoder_2)
        except ImportError:
            logging.warning("optimum.quanto not available, skipping T5 quantization")

    cache_args = {
        "use_teacache": engine_args.use_teacache,
        "use_fbcache": engine_args.use_fbcache,
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": input_config.num_inference_steps,
    }

    if local_rank == world_size - 1:
        gpu_indices = list(range(world_size))
        monitor = ZeusMonitor(gpu_indices=gpu_indices)

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        cache_args=cache_args,
        torch_dtype=torch.bfloat16,
        text_encoder_2=text_encoder_2,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    # Print batch information
    if args.use_dataset:
        print(f"Starting batch inference with {len(input_config.prompt)} prompts from dataset...")
    else:
        batch_size = len(input_config.prompt) if isinstance(input_config.prompt, list) else 1
        print(f"Starting inference with batch size {batch_size}...")

    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    torch.distributed.barrier()
    start_time = time.time()
    if local_rank == world_size - 1:
        monitor.begin_window("step")

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=256,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )

    torch.cuda.synchronize()
    torch.distributed.barrier()
    end_time = time.time()
    if local_rank == world_size - 1:
        result = monitor.end_window("step")
    
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                
                # Create filename with prompt info if using dataset
                if args.use_dataset and isinstance(input_config.prompt, list) and image_rank < len(input_config.prompt):
                    prompt_snippet = input_config.prompt[image_rank][:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
                    image_name = f"flux_dataset_{parallel_info}_{image_rank}_{prompt_snippet}_tc_{engine_args.use_torch_compile}.png"
                    print(f"Image {i} saved to ./results/{image_name}")
                    print(f"  Prompt: {input_config.prompt[image_rank]}")
                else:
                    image_name = f"flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                    print(f"Image {i} saved to ./results/{image_name}")
                
                image.save(f"./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        batch_size = len(input_config.prompt) if isinstance(input_config.prompt, list) else 1
        print(f"\n=== Inference Results ===")
        print(f"Batch size: {batch_size}")
        print(f"Total time: {elapsed_time:.2f} sec")
        print(f"Time per image: {elapsed_time/batch_size:.2f} sec") 
        print(f"Parameter memory: {parameter_peak_memory/1e9:.2f} GB")
        print(f"Peak memory: {peak_memory/1e9:.2f} GB")
        print(f"Throughput: {batch_size/elapsed_time:.2f} images/sec")
    
        print(f"Zeus time: {result.time} (s)")
        print(f"total energy: {result.total_energy} (J)")
        for i in range(world_size):
            print(f"rank {i} energy: {result.gpu_energy[i]} (J)")
        
        with open(f"./results/flux.csv", "a") as f:
            line_str = f"{batch_size},{input_config.height},{input_config.width}"
            line_str += f",{input_config.num_inference_steps}"
            line_str += f",{engine_args.ulysses_degree},{engine_args.ring_degree}"
            line_str += f",{engine_args.use_torch_compile}"
            line_str += f",{result.time},{result.total_energy}"
            for i in range(world_size):
                line_str += f",{result.gpu_energy[i]}"
            line_str += "\n"
            f.write(line_str)

    # pid = os.getpid()
    # print(f"Killing process group {pid}")
    # os.system(f'pkill -P {pid}')
        
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main() 