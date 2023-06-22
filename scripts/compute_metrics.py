import os
import csv

import tyro
import pandas as pd


def main(data_dir: str, out_file: str) -> None:
    """Compute metrics for all models in the given directory."""
    model_names = os.listdir(data_dir)
    print(f"{model_names=}")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out_csv = csv.writer(open(out_file, "w", newline=""))
    metrics = ["throughput", "response_length", "latency", "energy"]
    out_csv.writerow(["model"] + metrics)

    for model_name in model_names:
        df = pd.read_json(f"{data_dir}/{model_name}/benchmark.json")
        out_csv.writerow(
            [model_name.replace("--", "/")] + df[metrics].mean().to_list(),
        )


if __name__ == "__main__":
    tyro.cli(main)
