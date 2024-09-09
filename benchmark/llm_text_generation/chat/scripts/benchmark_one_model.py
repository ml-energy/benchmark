from __future__ import annotations

import os
import argparse
import subprocess
from itertools import product


def print_and_write(outfile, line: str, flush: bool = False):
    print(line, end="", flush=flush)
    outfile.write(line)
    if flush:
        outfile.flush()


def main(args: argparse.Namespace) -> None:
    hf_token = os.environ["HF_TOKEN"]

    outdir = f"{args.result_root}/{args.model}"
    os.makedirs(outdir, exist_ok=True)

    outfile = open(f"{outdir}/gpus{''.join(args.gpu_ids)}.out.txt", "a")

    assert len(args.backends) == len(args.server_images)
    server_images = dict(zip(args.backends, args.server_images))

    print_and_write(outfile, f"Benchmarking {args.model}\n")
    print_and_write(outfile, f"Backends: {args.backends}\n")
    print_and_write(outfile, f"Server images: {args.server_images}\n")
    print_and_write(outfile, f"Request rates: {args.request_rates}\n")
    print_and_write(outfile, f"Power limits: {args.power_limits}\n")
    print_and_write(outfile, f"Max number of seqs: {args.max_num_seqs}\n")

    for backend, request_rate, power_limit, max_num_seqs in product(args.backends, args.request_rates, args.power_limits, args.max_num_seqs):
        print_and_write(outfile, f"{backend=}, {request_rate=}, {power_limit=}, {max_num_seqs=}\n", flush=True)
        with subprocess.Popen(
            args=[
                "python",
                "scripts/benchmark_one_datapoint.py",
                "--backend", backend,
                "--server-image", server_images[backend],
                "--model", args.model,
                "--sharegpt-path", "sharegpt/ShareGPT_V3_filtered_500.json",
                "--request-rate", request_rate,
                "--power-limit", power_limit,
                "--max-num-seqs", max_num_seqs,
                "--result-root", args.result_root,
                "--huggingface-token", hf_token,
                "--gpu-ids", *args.gpu_ids,
                "--log-level", "INFO",
                "--data-dup-factor", args.data_dup_factor,
                "--nnodes", args.nnodes,
                "--node-id", args.node_id,
                "--head-node-address", args.head_node_address,
                "--preemption-mode", args.preemption_mode,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc:
            if proc.stdout:
                i = 0
                for line in proc.stdout:
                    print_and_write(outfile, line, flush=i % 50 == 0)
                    i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="ID of the model to benchmark")
    parser.add_argument("--result-root", type=str, help="Root directory to store the results")
    parser.add_argument("--gpu-ids", type=str, nargs="+", help="GPU IDs to use")
    parser.add_argument("--backends", type=str, nargs="+", default=["vllm", "tgi"], help="Backends to benchmark")
    parser.add_argument("--server-images", type=str, nargs="+", default=["mlenergy/vllm:v0.4.2-openai", "mlenergy/tgi:v2.0.2"], help="Server images to benchmark")
    parser.add_argument("--request-rates", type=str, nargs="+", default=["4.00", "3.00", "2.00", "1.00"], help="Request rates to benchmark")
    parser.add_argument("--power-limits", type=str, nargs="+", default=["400", "300", "200"], help="Power limits to benchmark")
    parser.add_argument("--max-num-seqs", type=str, nargs="+", help="vLLM --max-num-seqs to benchmark")
    parser.add_argument("--data-dup-factor", type=str, default="1", help="How many times to repeat the ShareGPT dataset to generate more requests.")
    parser.add_argument("--nnodes", default="1", help="Number of nodes in the cluster.")
    parser.add_argument("--node-id", default="0", help="ID of the node the script was launched on.")
    parser.add_argument("--head-node-address", default="localhost", help="Address of the Ray head node.")
    parser.add_argument("--preemption-mode", default="recompute", help="vLLM engine preemption mode.")
    args = parser.parse_args()
    main(args)
