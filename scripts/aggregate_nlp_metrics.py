import os
import json

import tyro
import pandas as pd

TASK_METRICS = {
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "truthfulqa_mc": "mc2",
}

TASK_SHORT_NAMES = {
    "arc_challenge": "arc",
    "hellaswag": "hellaswag",
    "truthfulqa_mc": "truthfulqa",
}


def main(data_dir: str, out_file: str = "score.csv") -> None:
    """Aggregate results from lm-evaluation-harness into a CSV file.

    Args:
        data_dir: The directory containing the results. Model names are
            expected to be the immediate subdirectories of `data_dir`.
        out_file: The path to the output CSV file. (Default: `score.csv`)
    """
    models = list(filter(lambda x: os.path.isdir(f"{data_dir}/{x}"), os.listdir(data_dir)))

    df = pd.DataFrame(columns=TASK_SHORT_NAMES.values())
    for model_dir in models:
        for task, metric in TASK_METRICS.items():
            model_name = "/".join(model_dir.split("--")[-2:])
            results = json.load(open(f"{data_dir}/{model_dir}/{task}.json"))
            df.loc[model_name, TASK_SHORT_NAMES[task]] = float(results["results"][task][metric]) * 100.0
    df = df.reset_index().rename(columns={"index": "model"})

    # Write the CSV file.
    if dirname := os.path.dirname(out_file):
        os.makedirs(dirname, exist_ok=True)
    df.to_csv(out_file, index=False)

if __name__ == "__main__":
    tyro.cli(main)
