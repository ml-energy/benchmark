import os
import json
from contextlib import suppress

import tyro


def main(data_dir: str) -> None:
    """Summarize the results collected for all models in the given directory."""
    model_names = os.listdir(data_dir)
    print(len(model_names), "models found")

    for i, model_name in enumerate(model_names):
        try:
            benchmark = json.load(open(f"{data_dir}/{model_name}/benchmark.json"))
            print(f"{i:2d} {len(benchmark):5d} results found for", model_name)
        except json.JSONDecodeError:
            print(f"{i:2d} [ERR] results found for {model_name}")


if __name__ == "__main__":
    tyro.cli(main)
