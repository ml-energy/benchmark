import json
from glob import glob
from pathlib import Path

import tyro


FIELDS = {
    "model": "Model",
    "gpu_model": "GPU",
    "energy_per_image": "Energy/image (J)",
    "average_batch_latency": "Batch latency (s)",
    "batch_size": "Batch size",
    "num_inference_steps": "Denoising steps",
}

def main(results_dir: Path, output_dir: Path) -> None:
    print(f"{results_dir} -> {output_dir}")

    for model_dir in sorted(glob(f"{results_dir}/*/*")):
        model_name = "/".join(model_dir.split("/")[-2:])
        print(f"  {model_name}")
        (output_dir / model_name).mkdir(parents=True, exist_ok=True)
        for file in sorted(glob(f"{model_dir}/bs*+results.json")):
            raw_data = json.load(open(file))
            raw_data["energy_per_image"] = raw_data["average_batch_energy"] / raw_data["batch_size"]

            data = {}
            for field1, field2 in FIELDS.items():
                data[field2] = raw_data.pop(field1)

            filename = f"bs{data['Batch size']}+steps{data['Denoising steps']}.json"
            json.dump(data, open(output_dir / model_name/  filename, "w"), indent=2)


if __name__ == "__main__":
    tyro.cli(main)
