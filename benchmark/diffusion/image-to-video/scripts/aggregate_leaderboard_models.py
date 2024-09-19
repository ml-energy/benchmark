import json
from glob import glob
from pathlib import Path

import tyro

def raw_params_to_readable(params: int) -> str:
    return f"{params/1e9:.1f}B"

def main(results_dir: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"{results_dir} -> {output_file}")

    models = {}
    for model_dir in sorted(glob(f"{results_dir}/*/*")):
        model_name = "/".join(model_dir.split("/")[-2:])
        print(f"  {model_name}")
        result_file_cand = glob(f"{model_dir}/bs1+*+steps25+results.json")
        assert len(result_file_cand) == 1, model_name
        results_data = json.load(open(result_file_cand[0]))
        denosing_module_name = "unet" if "unet" in results_data["num_parameters"] else "transformer"
        model_info = dict(
            url=f"https://huggingface.co/{model_name}",
            nickname=model_name.split("/")[-1].replace("-", " ").title(),
            total_params=raw_params_to_readable(sum(results_data["num_parameters"].values())),
            denoising_params=raw_params_to_readable(results_data["num_parameters"][denosing_module_name]),
            resolution="NA",
        )
        assert model_name not in models
        models[model_name] = model_info

    json.dump(models, open(output_file, "w"), indent=2)


if __name__ == "__main__":
    tyro.cli(main)
