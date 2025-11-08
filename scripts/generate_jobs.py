"""Generate job files (Pegasus queue.yaml or Slurm scripts) from configs directory.

This script scans the configs/vllm directory structure and generates job files
based on benchmark.yaml (per task) and num_gpus.txt (per model & GPU).
"""

from __future__ import annotations

import yaml
import tyro
import dataclasses
from pathlib import Path
from pydantic import BaseModel
from collections import defaultdict
from dataclasses import dataclass


class BenchmarkTemplate(BaseModel):
    """Benchmark template from benchmark.yaml."""

    command_template: str
    sweeps: dict[str, list[int]]


class ModelWorkload(BaseModel):
    """Workload configuration for a single model."""

    model_id: str
    config_dir: Path
    max_num_seqs: list[int]


@dataclass
class DatasetConfig:
    """Configuration for a dataset including its template and workloads."""

    template: BenchmarkTemplate
    workloads: dict[str, dict[int, list[ModelWorkload]]]


@dataclass
class Pegasus:
    """Configuration for Pegasus queue.yaml generation."""

    pass


@dataclass
class Slurm:
    """Slurm-specific configuration options."""

    partition: str | None = None
    """Slurm partition name"""

    time_limit: str | None = None
    """Slurm time limit in hours:minutes:seconds (e.g., 48:00:00)"""

    cpus_per_gpu: int | None = None
    """CPUs per GPU for proportional allocation"""

    mem_per_gpu: str | None = None
    """Memory per GPU (e.g., 80G, 256000M)"""


@dataclass
class Generate[OutputConfigT: (Pegasus, Slurm)]:
    """Main configuration for the job generator."""

    output_dir: Path
    """Output directory for generated files"""

    output: OutputConfigT
    """Output-specific configuration (Pegasus or Slurm)"""

    configs_dir: Path = Path("configs")
    """Path to configs directory"""

    datasets: list[str] = dataclasses.field(default_factory=list)
    """Filter by specific datasets"""

    gpu_models: list[str] = dataclasses.field(default_factory=list)
    """Filter by specific GPU models"""


def load_benchmark_template(dataset_dir: Path) -> BenchmarkTemplate:
    """Load benchmark.yaml template for a dataset."""
    template_file = dataset_dir / "benchmark.yaml"
    if not template_file.exists():
        raise FileNotFoundError(f"benchmark.yaml not found in {dataset_dir}")

    with open(template_file) as f:
        data = yaml.safe_load(f)

    return BenchmarkTemplate(**data)


def scan_configs(configs_dir: Path) -> dict[str, DatasetConfig]:
    """Scan configs directory and return workload information."""
    configs_vllm = configs_dir / "vllm"
    if not configs_vllm.exists():
        raise ValueError(f"Directory not found: {configs_vllm}")

    datasets: dict[str, DatasetConfig] = {}

    for dataset_dir in sorted(configs_vllm.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        try:
            template = load_benchmark_template(dataset_dir)
        except FileNotFoundError:
            print(f"Skipping {dataset}: no benchmark.yaml found")
            continue

        dataset_config = DatasetConfig(
            template=template, workloads=defaultdict(lambda: defaultdict(list))
        )

        for org_dir in sorted(dataset_dir.iterdir()):
            if not org_dir.is_dir():
                continue

            for model_dir in sorted(org_dir.iterdir()):
                if not model_dir.is_dir():
                    continue

                model_id = f"{org_dir.name}/{model_dir.name}"

                for gpu_dir in sorted(model_dir.iterdir()):
                    if not gpu_dir.is_dir():
                        continue

                    gpu_model = gpu_dir.name
                    num_gpus_file = gpu_dir / "num_gpus.txt"

                    if not num_gpus_file.exists():
                        print(f"Warning: {num_gpus_file} not found, skipping")
                        continue

                    with open(num_gpus_file) as f:
                        gpu_counts = [int(line.strip()) for line in f if line.strip()]

                    for num_gpus in gpu_counts:
                        workload = ModelWorkload(
                            model_id=model_id,
                            config_dir=gpu_dir,
                            max_num_seqs=template.sweeps["max_num_seqs"],
                        )
                        dataset_config.workloads[gpu_model][num_gpus].append(workload)

        datasets[dataset] = dataset_config

    return datasets


def format_command(
    template: str, model_id: str, gpu_model: str, max_num_seqs_placeholder: str
) -> str:
    """Format command template with model_id and gpu_model."""
    return template.format(
        model_id=model_id, gpu_model=gpu_model, max_num_seqs=max_num_seqs_placeholder
    )


def slugify(s: str) -> str:
    """Convert string to filesystem-safe slug."""
    return s.replace("/", "_").replace(" ", "_")


def flatten_command(command: str) -> str:
    """Flatten a multiline command into a single line."""
    # Remove line continuation backslashes and newlines
    flattened = command.replace("\\\n", " ").replace("\n", " ")
    # Collapse multiple spaces into single spaces
    import re

    flattened = re.sub(r"\s+", " ", flattened)
    return flattened.strip()


def generate_pegasus_queues(
    all_workloads: dict[str, DatasetConfig],
    output_dir: Path,
    config: Pegasus,
) -> list[Path]:
    """Generate Pegasus queue.yaml files, one per GPU count."""
    # Organize workloads by GPU count
    by_gpu_count: dict[int, list[tuple[str, str, ModelWorkload, BenchmarkTemplate]]] = (
        defaultdict(list)
    )

    for dataset, dataset_config in all_workloads.items():
        for gpu_model, gpu_workloads in dataset_config.workloads.items():
            for num_gpus, workloads in gpu_workloads.items():
                for workload in workloads:
                    by_gpu_count[num_gpus].append(
                        (dataset, gpu_model, workload, dataset_config.template)
                    )

    output_files = []

    for num_gpus, jobs in sorted(by_gpu_count.items()):
        queue_data = []

        for dataset, gpu_model, workload, template in sorted(
            jobs, key=lambda x: (x[0], x[1], x[2].model_id)
        ):
            command = format_command(
                template.command_template,
                workload.model_id,
                gpu_model,
                max_num_seqs_placeholder="{{ max_num_seqs }}",
            )
            # Flatten multiline command to single line
            command = flatten_command(command)

            queue_data.append(
                {
                    "command": [command],
                    "max_num_seqs": workload.max_num_seqs,
                }
            )

        output_file = output_dir / f"queue_{num_gpus}gpu.yaml"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            yaml.dump(
                queue_data, f, default_flow_style=False, sort_keys=False, width=1000
            )

        output_files.append(output_file)
        print(f"Generated {output_file} with {len(queue_data)} job(s)")

    return output_files


def generate_slurm_script(
    dataset: str,
    gpu_model: str,
    num_gpus: int,
    workload: ModelWorkload,
    template: BenchmarkTemplate,
    output_dir: Path,
    slurm_config: Slurm,
) -> Path:
    """Generate a Slurm script for a single model."""
    model_slug = slugify(workload.model_id)
    output_file = output_dir / f"{dataset}_{gpu_model}_{num_gpus}gpu_{model_slug}.sh"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    script_lines = [
        "#!/bin/bash",
        "",
        f"# Slurm script for {dataset} / {workload.model_id} / {gpu_model} / {num_gpus} GPU{'' if num_gpus == 1 else 's'}",
        "",
    ]

    if slurm_config.partition:
        script_lines.append(f"#SBATCH --partition={slurm_config.partition}")
    if slurm_config.time_limit:
        script_lines.append(f"#SBATCH --time={slurm_config.time_limit}")

    script_lines.append("#SBATCH --ntasks=1")
    script_lines.append(f"#SBATCH --gpus-per-task={num_gpus}")

    if slurm_config.cpus_per_gpu:
        script_lines.append(f"#SBATCH --cpus-per-gpu={slurm_config.cpus_per_gpu}")
    if slurm_config.mem_per_gpu:
        script_lines.append(f"#SBATCH --mem-per-gpu={slurm_config.mem_per_gpu}")

    script_lines.extend(
        [
            f"#SBATCH --job-name={dataset}_{model_slug}",
            f"#SBATCH --output=logs/{dataset}_{gpu_model}_{num_gpus}gpu_{model_slug}_%j.out",
            f"#SBATCH --error=logs/{dataset}_{gpu_model}_{num_gpus}gpu_{model_slug}_%j.err",
            "",
            "# Environment variables (set these before running sbatch or export before submission)",
            "# export HF_TOKEN=<your_token>",
            "",
            "set -e",
            "",
            "# Change to submission directory",
            "cd $SLURM_SUBMIT_DIR",
            "",
            "# Temporary HF_HOME for this model only.",
            f"export HF_HOME=$SLURM_SUBMIT_DIR/hf_home/{model_slug}",
            "mkdir -p $HF_HOME",
            "",
            "# Cleanup on exit (success or failure)",
            'trap "rm -rf $HF_HOME" EXIT',
            "",
        ]
    )

    # Generate command template with $max_num_seqs variable
    command_template_str = format_command(
        template.command_template,
        workload.model_id,
        gpu_model,
        max_num_seqs_placeholder="$max_num_seqs",
    )

    # Create for loop over max_num_seqs values
    max_num_seqs_list = " ".join(str(x) for x in workload.max_num_seqs)
    script_lines.extend(
        [
            f"for max_num_seqs in {max_num_seqs_list}; do",
            '  echo "Running with max-num-seqs=$max_num_seqs"',
            f"  {command_template_str}",
            "done",
        ]
    )

    with open(output_file, "w") as f:
        f.write("\n".join(script_lines))

    output_file.chmod(0o755)

    return output_file


def generate_slurm_scripts(
    all_workloads: dict[str, DatasetConfig],
    output_dir: Path,
    slurm_config: Slurm,
) -> list[Path]:
    """Generate Slurm scripts for all workloads."""
    output_files = []

    for dataset, dataset_config in sorted(all_workloads.items()):
        for gpu_model, gpu_workloads in sorted(dataset_config.workloads.items()):
            for num_gpus, workloads in sorted(gpu_workloads.items()):
                for workload in sorted(workloads, key=lambda w: w.model_id):
                    output_file = generate_slurm_script(
                        dataset,
                        gpu_model,
                        num_gpus,
                        workload,
                        dataset_config.template,
                        output_dir,
                        slurm_config,
                    )
                    print(f"  Generated {output_file}")
                    output_files.append(output_file)

    return output_files


def main(config: Generate[Pegasus] | Generate[Slurm]) -> None:
    """Generate job files from config directory."""
    print(f"Scanning configs in: {config.configs_dir}")
    print(f"Output directory: {config.output_dir}")
    print()

    datasets = scan_configs(config.configs_dir)

    if not datasets:
        print("No datasets found with benchmark.yaml files")
        return

    # Apply filters
    filtered_datasets = {}
    for dataset, dataset_config in datasets.items():
        if config.datasets and dataset not in config.datasets:
            continue

        filtered_config = DatasetConfig(template=dataset_config.template, workloads={})

        for gpu_model, gpu_workloads in dataset_config.workloads.items():
            if config.gpu_models and gpu_model not in config.gpu_models:
                continue
            filtered_config.workloads[gpu_model] = gpu_workloads

        if filtered_config.workloads:
            filtered_datasets[dataset] = filtered_config

    match config.output:
        case Pegasus():
            output_files = generate_pegasus_queues(
                filtered_datasets, config.output_dir, config.output
            )
        case Slurm():
            output_files = generate_slurm_scripts(
                filtered_datasets, config.output_dir, config.output
            )
        case _:
            raise ValueError("Unsupported output configuration")

    print(f"\nGenerated {len(output_files)} output file(s).")


if __name__ == "__main__":
    main(tyro.cli(Generate[Pegasus] | Generate[Slurm]))
