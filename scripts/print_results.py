import os
import json

import tyro


def main(data_dir: str, depth: int = 1) -> None:
    """Summarize the results collected for all models in the given directory.
    
    Args:
        data_dir: The directory containing the results.
        depth: The depth of the directory tree to search. When it's 1, the
            script expects to fine model directories directly under `data_dir`.
            (Default: 1)
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    if depth == 1:
        model_names = os.listdir(data_dir)
        print(len(model_names), "models found in", data_dir)

        for i, model_name in enumerate(model_names):
            if not os.path.isdir(f"{data_dir}/{model_name}"):
                continue
            try:
                benchmark = json.load(open(f"{data_dir}/{model_name}/benchmark.json"))
                print(f"{i:2d} {len(benchmark):5d} results found for", model_name)
            except json.JSONDecodeError:
                print(f"{i:2d} [ERR] results found for {model_name}")

    else:
        for dir in os.listdir(data_dir):
            main(f"{data_dir}/{dir}", depth - 1)


if __name__ == "__main__":
    tyro.cli(main)
