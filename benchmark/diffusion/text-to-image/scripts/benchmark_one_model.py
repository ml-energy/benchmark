from __future__ import annotations

import os
import argparse
import subprocess


def print_and_write(outfile, line: str, flush: bool = False):
    print(line, end="", flush=flush)
    outfile.write(line)
    if flush:
        outfile.flush()


def main(args: argparse.Namespace) -> None:
    assert len(args.gpu_ids) == 1

    hf_token = os.environ["HF_TOKEN"]

    if args.model.startswith("models/"):
        outdir = f"{args.result_root}/{args.model[len('models/'):]}"
    else:
        outdir = f"{args.result_root}/{args.model}"
    os.makedirs(outdir, exist_ok=True)

    outfile = open(f"{outdir}/gpus{''.join(args.gpu_ids)}.out.txt", "w")

    print_and_write(outfile, f"Benchmarking {args.model}\n")
    print_and_write(outfile, f"Batch sizes: {args.batch_sizes}\n")
    print_and_write(outfile, f"Power limits: {args.power_limits}\n")

    for batch_size in args.batch_sizes:
        for power_limit in args.power_limits:
            print_and_write(outfile, f"{batch_size=}, {power_limit=}\n", flush=True)
            with subprocess.Popen(
                args=[
                    "docker", "run",
                    "--gpus", '"device=' + ','.join(args.gpu_ids) + '"',
                    "--cap-add", "SYS_ADMIN",
                    "--name", f"leaderboard-t2i-{''.join(args.gpu_ids)}",
                    "--rm",
                    "-v", "/data/leaderboard/hfcache:/root/.cache/huggingface",
                    "-v", f"{os.getcwd()}:/workspace/text-to-image",
                    "mlenergy/leaderboard:diffusion-t2i",
                    "--result-root", args.result_root,
                    "--batch-size", batch_size,
                    "--num-batches", "10",
                    "--power-limit", power_limit,
                    "--model", args.model,
                    "--huggingface-token", hf_token,
                    "--num-inference-steps", "25",
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

            # If proc exited with non-zero status, it's probably an OOM.
            # Move on to the next batch size.
            if proc.returncode != 0:
                break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="ID of the model to benchmark")
    parser.add_argument("--result-root", type=str, help="Root directory to store the results")
    parser.add_argument("--gpu-ids", type=str, nargs="+", help="GPU IDs to use")
    parser.add_argument("--batch-sizes", type=str, nargs="+", default=["8", "4", "2", "1"], help="Batch sizes to benchmark")
    parser.add_argument("--power-limits", type=str, nargs="+", default=["400", "300", "200"], help="Power limits to benchmark")
    args = parser.parse_args()
    main(args)
