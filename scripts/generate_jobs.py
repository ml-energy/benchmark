"""Generate job files (Pegasus queue.yaml or Slurm scripts) from configs directory.

This script scans the configs/vllm directory structure and generates job files
based on benchmark.yaml (per task) and num_gpus.txt (per model & GPU).
"""

from __future__ import annotations

import re
import yaml
import tyro
import dataclasses
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from collections import defaultdict
from dataclasses import dataclass
from itertools import product


class BenchmarkTemplate(BaseModel):
    """Benchmark template from benchmark.yaml."""

    command_template: str
    sweep_defaults: list[dict[str, list[Any]]]


class SweepConfig(BaseModel):
    """Sweep configuration from sweeps.yaml."""

    sweep: list[dict[str, list[Any]]]


class ModelWorkload(BaseModel):
    """Workload configuration for a single model."""

    model_id: str
    config_dir: Path
    sweep_combinations: list[dict[str, Any]]


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

    account: str | None = None
    """Slurm account for billing"""


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


def extract_template_placeholders(template: str) -> set[str]:
    """Extract all {placeholder} names from a command template.

    Args:
        template: Command template string with {placeholder} variables

    Returns:
        Set of placeholder names (without braces)
    """
    return set(re.findall(r"\{(\w+)\}", template))


def validate_sweep_keys(
    sweep_config: list[dict[str, list[Any]]],
    template: str,
    config_source: str,
) -> None:
    """Validate that sweep parameter keys exist in the command template.

    Args:
        sweep_config: List of sweep parameter dicts
        template: Command template string
        config_source: Description of config source (for error messages)
    """
    template_placeholders = extract_template_placeholders(template)

    # Collect all sweep parameter names
    sweep_params = set()
    for param_group in sweep_config:
        sweep_params.update(param_group.keys())

    # Check that all sweep params exist in template
    missing_in_template = sweep_params - template_placeholders
    if missing_in_template:
        raise ValueError(
            f"{config_source}: Sweep parameters {missing_in_template} "
            f"not found in command template. Available placeholders: {template_placeholders}"
        )


def validate_all_placeholders_filled(
    template: str,
    params: dict[str, Any],
    config_source: str,
) -> None:
    """Validate that all template placeholders will be filled.

    Args:
        template: Command template string
        params: Parameters to fill the template
        config_source: Description of config source (for error messages)
    """
    template_placeholders = extract_template_placeholders(template)
    param_keys = set(params.keys())

    missing_params = template_placeholders - param_keys
    if missing_params:
        raise ValueError(
            f"{config_source}: Template placeholders {missing_params} "
            f"not provided in parameters. Available params: {param_keys}"
        )


def compute_cartesian_product(
    param_group: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Compute Cartesian product of parameter lists in a single group.

    Args:
        param_group: Dict mapping parameter names to lists of values

    Returns:
        List of parameter combinations (one dict per combination)

    Example:
        >>> compute_cartesian_product({'a': [1, 2], 'b': [3, 4]})
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    if not param_group:
        return [{}]

    keys = list(param_group.keys())
    values = [param_group[key] for key in keys]

    combinations = []
    for value_tuple in product(*values):
        combinations.append(dict(zip(keys, value_tuple)))

    return combinations


def compute_sweep_combinations(
    sweep_config: list[dict[str, list[Any]]],
) -> list[dict[str, Any]]:
    """Compute all sweep combinations by flattening Cartesian products.

    Args:
        sweep_config: List of parameter group dicts

    Returns:
        Flattened list of all parameter combinations

    Example:
        >>> compute_sweep_combinations([
        ...     {'a': [1, 2], 'b': [3]},
        ...     {'a': [10], 'b': [30, 40]}
        ... ])
        [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}, {'a': 10, 'b': 30}, {'a': 10, 'b': 40}]
    """
    all_combinations = []
    for param_group in sweep_config:
        all_combinations.extend(compute_cartesian_product(param_group))
    return all_combinations


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

                    # Check for model+GPU-specific sweeps.yaml
                    sweeps_file = gpu_dir / "sweeps.yaml"
                    if sweeps_file.exists():
                        with open(sweeps_file) as f:
                            sweep_data = yaml.safe_load(f)
                        sweep_config_obj = SweepConfig(**sweep_data)
                        sweep_config = sweep_config_obj.sweep
                        config_source = f"sweeps.yaml in {gpu_dir}"
                    else:
                        # Fall back to task-level sweep_defaults
                        sweep_config = template.sweep_defaults
                        config_source = f"benchmark.yaml for {dataset}"

                    # Validate sweep configuration
                    validate_sweep_keys(
                        sweep_config, template.command_template, config_source
                    )

                    # Compute all sweep combinations
                    sweep_combinations = compute_sweep_combinations(sweep_config)

                    for num_gpus in gpu_counts:
                        workload = ModelWorkload(
                            model_id=model_id,
                            config_dir=gpu_dir,
                            sweep_combinations=sweep_combinations,
                        )
                        dataset_config.workloads[gpu_model][num_gpus].append(workload)

        datasets[dataset] = dataset_config

    return datasets


def format_command(
    template: str,
    params: dict[str, Any],
    config_source: str = "",
) -> str:
    """Format command template with all parameters.

    Args:
        template: Command template string with {placeholder} variables
        params: Dictionary of all parameter values to substitute
        config_source: Description of config source (for error messages)

    Returns:
        Formatted command string

    Raises:
        ValueError: If any template placeholders are not provided in params
    """
    # Validate that all placeholders will be filled
    if config_source:
        validate_all_placeholders_filled(template, params, config_source)

    return template.format(**params)


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
            # Generate one queue entry per sweep combination
            for sweep_params in workload.sweep_combinations:
                # Build full parameter dict
                params = {
                    "model_id": workload.model_id,
                    "gpu_model": gpu_model,
                    **sweep_params,
                }

                # Format command with all parameters
                command = format_command(
                    template.command_template,
                    params,
                    config_source=f"Pegasus: {dataset}/{workload.model_id}/{gpu_model}",
                )
                # Flatten multiline command to single line
                command = flatten_command(command)

                queue_data.append({"command": [command]})

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
    if slurm_config.account:
        script_lines.append(f"#SBATCH --account={slurm_config.account}")
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

    # Pre-compute all sweep combinations
    sweep_combinations = workload.sweep_combinations

    if not sweep_combinations:
        raise ValueError(f"No sweep combinations found for {workload.model_id}")

    # Get parameter names from the first combination (all should have same keys)
    param_names = list(sweep_combinations[0].keys())

    # Generate readable comment showing combinations
    script_lines.append("# Sweep combinations (one per line):")
    for combo in sweep_combinations:
        combo_str = " ".join(f"{k}={v}" for k, v in combo.items())
        script_lines.append(f"# {combo_str}")
    script_lines.append("")

    # Build bash array of combinations
    script_lines.append("combinations=(")
    for combo in sweep_combinations:
        # Each combination as space-separated values
        values = " ".join(str(combo[k]) for k in param_names)
        script_lines.append(f'  "{values}"')
    script_lines.append(")")
    script_lines.append("")

    # Generate for loop that iterates over combinations
    read_vars = " ".join(param_names)
    script_lines.append('for combo in "${combinations[@]}"; do')
    script_lines.append(f'  read -r {read_vars} <<< "$combo"')

    # Generate echo statement showing current parameter values
    echo_parts = " ".join(f"{name}=${name}" for name in param_names)
    script_lines.append(f'  echo "Running with {echo_parts}"')

    # Build parameter dict for command formatting (using bash variables)
    bash_params = {
        "model_id": workload.model_id,
        "gpu_model": gpu_model,
    }
    for param_name in param_names:
        bash_params[param_name] = f"${param_name}"

    # Format command with bash variable references
    command_str = format_command(
        template.command_template,
        bash_params,
        config_source=f"Slurm: {dataset}/{workload.model_id}/{gpu_model}",
    )

    script_lines.append(f"  {command_str}")
    script_lines.append("done")

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
