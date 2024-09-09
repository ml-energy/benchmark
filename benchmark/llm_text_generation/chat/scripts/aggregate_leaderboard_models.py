import json
from glob import glob
from pathlib import Path

import tyro

def infer_params_from_name(model_name: str) -> str:
    """Try to guess the model parameters from the model name.

    It basically parses the model name to look for a string that looks like "%d[bB]".

    Examples:
        - "meta-llama/Meta-Llama-3.1-8B-Instruct" -> 8B
        - "facebook/chameleon-30b" -> 30B
    """
    for token in model_name.lower().split("-"):
        if token.endswith("b") and token[:-1].isdigit():
            return token.upper()
    return "NA"


def main(results_dir: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"{results_dir} -> {output_file}")

    models = {}
    for model_dir in sorted(glob(f"{results_dir}/*/*")):
        model_name = "/".join(model_dir.split("/")[-2:])
        print(f"  {model_name}")
        model_info = dict(
            url=f"https://huggingface.co/{model_name}",
            nickname=model_name.split("/")[-1].replace("-", " ").title(),
            params=infer_params_from_name(model_name),
        )
        assert model_name not in models
        models[model_name] = model_info

    json.dump(models, open(output_file, "w"), indent=2)


if __name__ == "__main__":
    tyro.cli(main)
