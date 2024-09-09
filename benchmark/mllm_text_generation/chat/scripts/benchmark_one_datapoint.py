from __future__ import annotations

import os
import json
import atexit
import argparse
import subprocess
from pathlib import Path
import time
from typing import Literal


def set_power_limit(power_limit: int, gpu_ids: list[int]) -> None:
    for gpu_id in gpu_ids:
        subprocess.check_call([
            "docker", "exec", "nvml",
            "nvidia-smi", "-i", str(gpu_id), "-pm", "1",
        ])
        subprocess.check_call([
            "docker", "exec", "nvml",
            "nvidia-smi", "-i", str(gpu_id), "-pl", str(power_limit),
        ])


def start_server(
    backend: Literal["vllm", "tgi"],
    server_image: str,
    port: int,
    model: str,
    huggingface_token: str,
    gpu_ids: list[int],
    log_level: str,
    result_root: str,
    benchmark_name: str,
    max_num_seqs: int,
) -> str:
    gpu_str = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    gpu_str = f'"device={gpu_str}"'
    hf_cache_path = "/data/leaderboard/hfcache"
    models_dir = f"{os.getcwd()}/models"
    tokconf_filename = f"{model}/tokenizer_config.json"
    tokconf_path = f"{models_dir}/{tokconf_filename}"
    revision_filename = f"{model}/revision.txt"
    revision_path = f"{models_dir}/{revision_filename}"
    container_name = f"leaderboard-{backend}-{''.join(str(gpu_id) for gpu_id in gpu_ids)}"
    dcgm_sock_path = "/var/run/nvidia-dcgm.sock"

    assert Path(hf_cache_path).exists(), f"Hugging Face cache not found: {hf_cache_path}"
    assert Path(tokconf_path).exists(), f"Tokenizer config not found: {tokconf_path}"
    assert Path(revision_path).exists(), f"Revision file not found: {revision_path}"

    if backend == "vllm":
        extra_docker_args = []
        if "google/gemma-2-" in model:
            extra_docker_args.extend(["-e", "VLLM_ATTENTION_BACKEND=FLASHINFER"])

        server_cmd = [
            "docker", "run",
            "--gpus", gpu_str,
            "--ipc", "host",
            "--name", container_name,
            "-e", f"HF_TOKEN={huggingface_token}",
            "-e", f"LOG_LEVEL={log_level}",
            "-e", f"RESULT_FILE_PREFIX=/results/{benchmark_name}",
            "-p", f"{port}:8000",
            "-v", f"{hf_cache_path}:/root/.cache/huggingface",
            "-v", f"{result_root}:/results",
            *extra_docker_args,
            server_image,
            "--model", model,
            "--revision", open(revision_path).read().strip(),
            "--chat-template", json.load(open(tokconf_path))["chat_template"],
            "--tensor-parallel-size", str(len(gpu_ids)),
            "--gpu-memory-utilization", "0.95",
            "--trust-remote-code",
            "--enable-chunked-prefill", "False",
            "--max-model-len", "4096",
            "--disable-frontend-multiprocessing",
            "--max-num-seqs", str(max_num_seqs),
        ]

    elif backend == "tgi":
        server_cmd = [
            "docker", "run",
            "--gpus", gpu_str,
            "--ipc", "host",
            "--name", container_name,
            "-e", f"HUGGING_FACE_HUB_TOKEN={huggingface_token}",
            "-e", f"LOG_LEVEL={log_level}",
            "-e", f"RESULT_FILE_PREFIX=/results/{benchmark_name}",
            "-p", f"{port}:80",
            "-v", f"{hf_cache_path}:/root/.cache/huggingface",
            "-v", f"{models_dir}:/models",
            "-v", f"{result_root}:/results",
            server_image,
            "--model-id", model,
            "--revision", open(revision_path).read().strip(),
            "--huggingface-hub-cache", "/root/.cache/huggingface/hub",
            "--tokenizer-config-path", f"/models/{tokconf_filename}",
            "--cuda-memory-fraction", "0.95",
            "--num-shard", str(len(gpu_ids)),
            "--max-concurrent-requests", "512",
            "--trust-remote-code",
        ]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if Path(dcgm_sock_path).exists():
        # Right after docker run.
        server_cmd.insert(2, f"{dcgm_sock_path}:{dcgm_sock_path}")
        server_cmd.insert(2, "-v")

    print("Server:", " ".join(server_cmd))
    subprocess.Popen(server_cmd)

    return container_name


def start_client(
    backend: Literal["vllm", "tgi"],
    port: int,
    model: str,
    dataset_path: str,
    request_rate: str,
    gpu_ids: list[int],
    benchmark_name: str,
    power_limit: int,
    max_num_seqs,
    data_dup_factor: int,
) -> subprocess.Popen:
    client_cmd = [
        "python", "scripts/benchmark_client.py",
        "--backend", backend,
        "--port", str(port),
        "--model", model,
        "--dataset-path", dataset_path,
        "--request-rate", request_rate,
        "--benchmark-name", benchmark_name,
        "--power-limit", str(power_limit),
        "--max-num-seqs", str(max_num_seqs),
        "--data-dup-factor", str(data_dup_factor),
    ]
    print("Client:", " ".join(client_cmd))
    return subprocess.Popen(
        client_cmd,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_ids)},
    )


def terminate_server(container_name: str) -> None:
    subprocess.run(["docker", "kill", "-s", "INT", container_name])
    subprocess.run(["timeout", "30", "docker", "wait", container_name])
    subprocess.run(["docker", "rm", "-f", container_name])


def main(args: argparse.Namespace) -> None:
    if args.model.startswith("models/"):
        args.model = args.model[len("models/"):]
    if args.model.endswith("/"):
        args.model = args.model[:-1]

    results_dir = Path(args.result_root) / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    benchmark_name = f"{args.backend}+rate{args.request_rate}+pl{args.power_limit}+maxbs{args.max_num_seqs}+gpus{''.join(str(i) for i in args.gpu_ids)}"

    port = 8000 + args.gpu_ids[0]

    server_handle = start_server(
        args.backend,
        args.server_image,
        port,
        args.model,
        args.huggingface_token,
        args.gpu_ids,
        args.log_level,
        str(results_dir.absolute()),
        benchmark_name,
        args.max_num_seqs,
    )
    kill_fn = lambda: terminate_server(server_handle)
    atexit.register(kill_fn)

    set_power_limit(args.power_limit, args.gpu_ids)

    client_handle = start_client(
        args.backend,
        port,
        args.model,
        args.dataset_path,
        args.request_rate,
        args.gpu_ids,
        str(results_dir / benchmark_name),
        args.power_limit,
        args.max_num_seqs,
        args.data_dup_factor,
    )

    try:
        exit_code = client_handle.wait(timeout=2 * 3600)
    except subprocess.TimeoutExpired:
        client_handle.terminate()
        raise RuntimeError("Benchmark client timed out after two hours")

    if exit_code != 0:
        raise RuntimeError(f"Benchmark client exited with code {exit_code}")

    terminate_server(server_handle)
    atexit.unregister(kill_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["vllm", "tgi"], help="Server to benchmark.")
    parser.add_argument("--server-image", required=True, help="Docker image to use for the server.")
    parser.add_argument("--model", required=True, help="Model to benchmark, e.g., meta-llama/Llama-2-7b-chat-hf.")
    parser.add_argument("--dataset-path", required=True, help="Path to the Llava Instruct dataset to feed to the server.")
    parser.add_argument("--request-rate", required=True, help="Poisson process rate for request arrival times.")
    parser.add_argument("--max-num-seqs", required=True, help="vLLM --max-num-seqs to use.")
    parser.add_argument("--power-limit", type=int, required=True, help="GPU power limit in Watts.")
    parser.add_argument("--result-root", default="results", help="Root directory to save results.")
    parser.add_argument("--huggingface-token", required=True, help="Hugging Face API token.")
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True, help="GPU IDs to use for the server.")
    parser.add_argument("--log-level", default="INFO", help="Logging level for the server.")
    parser.add_argument("--data-dup-factor", type=int, default=1, help="How many times to repeat the dataset to generate more requests.")
    main(parser.parse_args())
