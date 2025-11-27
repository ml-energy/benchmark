"""Generate job files (Pegasus queue.yaml or Slurm scripts) from configs directory.

This script scans the configs/vllm directory structure and generates job files
based on benchmark.yaml (per task) and num_gpus.txt (per model & GPU).
"""

from __future__ import annotations

import json
import re
import yaml
import tyro
import dataclasses
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel
from collections import defaultdict
from dataclasses import dataclass
from itertools import product


class BenchmarkTemplate(BaseModel):
    """Benchmark template from benchmark.yaml."""

    command_template: str
    sweep_defaults: list[dict[str, list[Any]]]
    workload_defaults: dict[str, Any] = {}


class SweepConfig(BaseModel):
    """Sweep configuration from sweeps.yaml."""

    sweep: list[dict[str, list[Any]]]


class ModelWorkload(BaseModel):
    """Workload configuration for a single model."""

    model_id: str
    config_dir: Path
    sweep_combinations: list[dict[str, Any]]
    workload_overrides: dict[str, Any] = {}
    runtime: Literal["vllm", "xdit"] = "vllm"


@dataclass
class DatasetConfig:
    """Configuration for a dataset including its template and workloads."""

    template: BenchmarkTemplate
    workloads: dict[str, dict[int, list[ModelWorkload]]]


@dataclass
class Pegasus:
    """Configuration for Pegasus queue.yaml generation.

    Attributes:
        gpus_per_node: Number of GPUs per node (used to generate hosts_*gpu.yaml files)
        hostname: Hostname to use in hosts files (default: "localhost")
    """

    gpus_per_node: int
    hostname: str = "localhost"


@dataclass
class Slurm:
    """Slurm-specific configuration options.

    Attributes:
        partition: Slurm partition name
        account: Slurm account for billing
        time_limit: Slurm time limit in hours:minutes:seconds (e.g., 48:00:00)
        cpus_per_gpu: CPUs per GPU for proportional allocation
        mem_per_gpu: Memory per GPU (e.g., 80G, 256000M)
    """

    partition: str | None = None
    account: str | None = None
    time_limit: str | None = None
    cpus_per_gpu: int | None = None
    mem_per_gpu: str | None = None


@dataclass
class Generate[OutputConfigT: (Pegasus, Slurm)]:
    """Main configuration for the job generator.

    Attributes:
        output_dir: Output directory for generated files
        output: Output-specific configuration (Pegasus or Slurm)
        configs_dir: Path to configs directory (default: "configs")
        datasets: Filter by specific datasets (default: all)
        gpu_models: Filter by specific GPU models (default: all)
        container_runtime: Container runtime ("docker" or "singularity"), or None
            for workloads that don't use containers (e.g., xDiT)
        server_image: Container image path (Docker image or .sif file path),
            or None for workloads that don't use containers
        override_sweeps: Optional JSON string for global sweep parameters override
        override_workload: Optional JSON string for global workload parameters override
    """

    output_dir: Path
    output: OutputConfigT
    configs_dir: Path = Path("configs")
    datasets: list[str] = dataclasses.field(default_factory=list)
    gpu_models: list[str] = dataclasses.field(default_factory=list)
    container_runtime: Literal["docker", "singularity"] | None = None
    server_image: str | None = None
    override_sweeps: str | None = None
    override_workload: str | None = None


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

    # Collect all sweep parameter names (excluding 'num_gpus' which is a filter key)
    sweep_params = set()
    for param_group in sweep_config:
        sweep_params.update(k for k in param_group.keys() if k != "num_gpus")

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


def filter_sweep_config_by_num_gpus(
    sweep_config: list[dict[str, list[Any]]],
    num_gpus: int,
) -> list[dict[str, list[Any]]]:
    """Filter sweep configuration groups by num_gpus.

    If a group has a 'num_gpus' key, it only applies to jobs with those GPU counts.
    If a group doesn't have 'num_gpus', it applies to all GPU counts.

    Args:
        sweep_config: List of parameter group dicts
        num_gpus: Number of GPUs for the current job

    Returns:
        Filtered sweep config with only applicable groups, with num_gpus key removed

    Example:
        >>> filter_sweep_config_by_num_gpus([
        ...     {'a': [1], 'b': [2]},
        ...     {'num_gpus': [2], 'a': [3], 'b': [4]}
        ... ], num_gpus=1)
        [{'a': [1], 'b': [2]}]
        >>> filter_sweep_config_by_num_gpus([
        ...     {'a': [1], 'b': [2]},
        ...     {'num_gpus': [2], 'a': [3], 'b': [4]}
        ... ], num_gpus=2)
        [{'a': [1], 'b': [2]}, {'a': [3], 'b': [4]}]
    """
    filtered_groups = []

    for param_group in sweep_config:
        # Check if this group has a num_gpus filter
        if "num_gpus" in param_group:
            # Only include this group if num_gpus matches
            if num_gpus in param_group["num_gpus"]:
                # Create a copy without the num_gpus key
                filtered_group = {
                    k: v for k, v in param_group.items() if k != "num_gpus"
                }
                filtered_groups.append(filtered_group)
        else:
            # No filter, applies to all num_gpus
            filtered_groups.append(param_group)

    return filtered_groups


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
    """Scan configs directory and return workload information.

    Supports multiple runtimes (vllm, xdit). Each runtime has different
    handling for GPU counts:
    - vllm: Uses num_gpus.txt and filter_sweep_config_by_num_gpus
    - xdit: Derives num_gpus from ulysses_degree * ring_degree
    """
    # Find available runtime directories
    runtime_roots: list[tuple[Path, Literal["vllm", "xdit"]]] = []
    for runtime_name in ("vllm", "xdit"):
        root = configs_dir / runtime_name
        if root.exists():
            runtime_roots.append((root, runtime_name))  # type: ignore[arg-type]

    if not runtime_roots:
        raise ValueError(
            f"No supported runtime directories found under: {configs_dir} "
            "(expected one of: vllm, xdit)"
        )

    datasets: dict[str, DatasetConfig] = {}

    for runtime_root, runtime_name in runtime_roots:
        for dataset_dir in sorted(runtime_root.iterdir()):
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

                        # Check for model+GPU-specific workload.yaml
                        workload_overrides = {}
                        workload_file = gpu_dir / "workload.yaml"
                        if workload_file.exists():
                            with open(workload_file) as f:
                                workload_overrides = yaml.safe_load(f) or {}

                        # Validate sweep configuration
                        validate_sweep_keys(
                            sweep_config, template.command_template, config_source
                        )

                        if runtime_name == "vllm":
                            # vLLM: Use num_gpus.txt and filter_sweep_config_by_num_gpus
                            num_gpus_file = gpu_dir / "num_gpus.txt"
                            if not num_gpus_file.exists():
                                print(f"Warning: {num_gpus_file} not found, skipping")
                                continue

                            with open(num_gpus_file) as f:
                                gpu_counts = [
                                    int(line.strip()) for line in f if line.strip()
                                ]

                            for num_gpus in gpu_counts:
                                # Filter sweep config for this specific num_gpus value
                                filtered_sweep_config = filter_sweep_config_by_num_gpus(
                                    sweep_config, num_gpus
                                )

                                # Compute sweep combinations for this num_gpus
                                sweep_combinations = compute_sweep_combinations(
                                    filtered_sweep_config
                                )

                                workload = ModelWorkload(
                                    model_id=model_id,
                                    config_dir=gpu_dir,
                                    sweep_combinations=sweep_combinations,
                                    workload_overrides=workload_overrides,
                                    runtime="vllm",
                                )
                                dataset_config.workloads[gpu_model][num_gpus].append(
                                    workload
                                )
                        else:
                            # xDiT: Derive num_gpus from ulysses_degree * ring_degree
                            base_combinations = compute_sweep_combinations(sweep_config)

                            # Group combinations by derived num_gpus
                            groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
                            for combo in base_combinations:
                                if (
                                    "ulysses_degree" not in combo
                                    or "ring_degree" not in combo
                                ):
                                    raise ValueError(
                                        f"Expected ulysses_degree and ring_degree in "
                                        f"sweep for xDiT task {dataset}, but they were "
                                        f"not found."
                                    )
                                ulysses_degree = int(combo["ulysses_degree"])
                                ring_degree = int(combo["ring_degree"])
                                num_gpus = ulysses_degree * ring_degree
                                combo_with_num = dict(combo)
                                combo_with_num["num_gpus"] = num_gpus
                                groups[num_gpus].append(combo_with_num)

                            for num_gpus, group_combos in groups.items():
                                workload = ModelWorkload(
                                    model_id=model_id,
                                    config_dir=gpu_dir,
                                    sweep_combinations=group_combos,
                                    workload_overrides=workload_overrides,
                                    runtime="xdit",
                                )
                                dataset_config.workloads[gpu_model][num_gpus].append(
                                    workload
                                )

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


def generate_pegasus_hosts(
    output_dir: Path,
    hostname: str,
    gpus_per_node: int,
    gpu_counts: list[int],
) -> list[Path]:
    """Generate hosts_*gpu.yaml files for Pegasus.

    Args:
        output_dir: Directory to write hosts files
        hostname: Hostname to use in hosts files
        gpus_per_node: Total number of GPUs per node
        gpu_counts: List of GPU counts to generate hosts files for

    Returns:
        List of generated hosts file paths
    """
    output_files = []

    for num_gpus in gpu_counts:
        # Generate CUDA_VISIBLE_DEVICES assignments
        # For example, with 8 GPUs per node and 2 GPUs per job:
        # ["0,1", "2,3", "4,5", "6,7"]
        cuda_devices = []
        num_slots = gpus_per_node // num_gpus

        for slot in range(num_slots):
            start_gpu = slot * num_gpus
            end_gpu = start_gpu + num_gpus
            gpu_list = ",".join(str(i) for i in range(start_gpu, end_gpu))
            cuda_devices.append(gpu_list)

        hosts_data = [
            {
                "hostname": [hostname],
                "cuda_visible_devices": cuda_devices,
            }
        ]

        output_file = output_dir / f"hosts_{num_gpus}gpu.yaml"
        with open(output_file, "w") as f:
            yaml.dump(hosts_data, f, default_flow_style=False, sort_keys=False)

        output_files.append(output_file)
        print(f"Generated {output_file} with {len(cuda_devices)} slot(s)")

    return output_files


def generate_pegasus_queues(
    all_workloads: dict[str, DatasetConfig],
    output_dir: Path,
    config: Pegasus,
    container_runtime: str | None,
    server_image: str | None,
    workload_override: dict[str, Any] | None = None,
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
                params: dict[str, Any] = {
                    "model_id": workload.model_id,
                    "gpu_model": gpu_model,
                    "num_gpus": num_gpus,
                    **template.workload_defaults,
                    **workload.workload_overrides,
                    **sweep_params,
                }
                # Add container params only for vLLM workloads when specified
                if workload.runtime == "vllm":
                    if container_runtime is not None:
                        params["container_runtime"] = container_runtime
                    if server_image is not None:
                        params["server_image"] = server_image
                if workload_override:
                    params.update(workload_override)

                # Format command with all parameters
                command = format_command(
                    template.command_template,
                    params,
                    config_source=f"Pegasus: {dataset}/{workload.model_id}/{gpu_model}",
                )
                # Flatten multiline command to single line
                command = flatten_command(command)

                # For xDiT: derive MASTER_PORT from cuda_visible_devices
                if workload.runtime == "xdit":
                    # Extract first GPU ID and compute port (8000 + first_gpu_id)
                    # Use export && to ensure variables are set before expansion
                    command = (
                        "export MASTER_PORT=$((8000 + $(echo {{ cuda_visible_devices }} | cut -d, -f1))) && "
                        "export MASTER_ADDR=$(hostname) && "
                        f"CUDA_VISIBLE_DEVICES={{{{ cuda_visible_devices }}}} {command}"
                    )
                else:
                    # Prepend CUDA_VISIBLE_DEVICES with Pegasus templating
                    command = (
                        f"CUDA_VISIBLE_DEVICES={{{{ cuda_visible_devices }}}} {command}"
                    )

                queue_data.append({"command": [command]})

        output_file = output_dir / f"queue_{num_gpus}gpu.yaml"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            yaml.dump(
                queue_data, f, default_flow_style=False, sort_keys=False, width=1000
            )

        output_files.append(output_file)
        print(f"Generated {output_file} with {len(queue_data)} job(s)")

    # Generate hosts files for all GPU counts
    gpu_counts = sorted(by_gpu_count.keys())
    hosts_files = generate_pegasus_hosts(
        output_dir, config.hostname, config.gpus_per_node, gpu_counts
    )
    output_files.extend(hosts_files)

    return output_files


def generate_vllm_slurm_script(
    dataset: str,
    gpu_model: str,
    num_gpus: int,
    workload: ModelWorkload,
    template: BenchmarkTemplate,
    output_dir: Path,
    slurm_config: Slurm,
    container_runtime: str | None,
    server_image: str | None,
    workload_override: dict[str, Any] | None = None,
) -> Path:
    """Generate a Slurm script for a vLLM workload."""
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

    script_lines.append(f"#SBATCH --gres=gpu:{num_gpus}")

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
            "set -v",
            "",
            "# Ensure required environment variables are set",
            'if [[ -z "$HF_HOME" ]]; then',
            '  echo "ERROR: HF_HOME environment variable is not set. Please export HF_HOME before running sbatch." >&2',
            "  exit 1",
            "fi",
            "",
            'if [[ ! -d "$HF_HOME" ]]; then',
            '  echo "ERROR: HF_HOME directory does not exist: $HF_HOME" >&2',
            "  exit 1",
            "fi",
            "",
            'if [[ -z "$HF_TOKEN" ]]; then',
            '  echo "ERROR: HF_TOKEN environment variable is not set. Please export HF_TOKEN before running sbatch." >&2',
            "  exit 1",
            "fi",
            "",
            "# Change to submission directory",
            "cd $SLURM_SUBMIT_DIR",
            "",
        ]
    )

    if container_runtime == "singularity":
        script_lines.append("# Load Singularity")
        script_lines.append("module load singularity || true")
        script_lines.append("")

    # Set CUDA_VISIBLE_DEVICES explicitly, although Slurm usually does this automatically
    gpu_ids = ",".join(str(i) for i in range(num_gpus))
    script_lines.append(f"export CUDA_VISIBLE_DEVICES={gpu_ids}")
    script_lines.append("")

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
    bash_params: dict[str, Any] = {
        "model_id": workload.model_id,
        "gpu_model": gpu_model,
        "num_gpus": num_gpus,
        **template.workload_defaults,
        **workload.workload_overrides,
    }
    if container_runtime is not None:
        bash_params["container_runtime"] = container_runtime
    if server_image is not None:
        bash_params["server_image"] = server_image
    if workload_override:
        bash_params.update(workload_override)
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


def generate_xdit_slurm_script(
    dataset: str,
    gpu_model: str,
    num_gpus: int,
    workload: ModelWorkload,
    template: BenchmarkTemplate,
    output_dir: Path,
    slurm_config: Slurm,
) -> Path:
    """Generate a Slurm script for an xDiT workload."""
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

    script_lines.append(f"#SBATCH --gres=gpu:{num_gpus}")

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
            "set -e",
            "",
            "# Change to submission directory",
            "cd $SLURM_SUBMIT_DIR",
            "",
            "# Load Python, CUDA, and GCC",
            "module load python/3.12.1 && \\",
            "module load cuda/12.6.3 && \\",
            "module load gcc",
            "",
            "source .venv/bin/activate",
            "",
            "# Ensure required environment variables are set",
            'if [[ -z "$HF_HOME" ]]; then',
            '  echo "ERROR: HF_HOME environment variable is not set. Please export HF_HOME before running sbatch." >&2',
            "  exit 1",
            "fi",
            "",
            'if [[ ! -d "$HF_HOME" ]]; then',
            '  echo "ERROR: HF_HOME directory does not exist: $HF_HOME" >&2',
            "  exit 1",
            "fi",
            "",
            'if [[ -z "$HF_TOKEN" ]]; then',
            '  echo "ERROR: HF_TOKEN environment variable is not set. Please export HF_TOKEN before running sbatch." >&2',
            "  exit 1",
            "fi",
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

    # Derive MASTER_PORT from first GPU in CUDA_VISIBLE_DEVICES
    script_lines.append("# Derive MASTER_PORT from first GPU ID (8000 + first_gpu_id)")
    script_lines.append(
        "MASTER_PORT=$((8000 + $(echo $CUDA_VISIBLE_DEVICES | cut -d, -f1)))"
    )
    script_lines.append("MASTER_ADDR=$(hostname)")
    script_lines.append("")

    # Generate for loop that iterates over combinations
    read_vars = " ".join(param_names)
    script_lines.append('for combo in "${combinations[@]}"; do')
    script_lines.append(f'  read -r {read_vars} <<< "$combo"')

    # Generate echo statement showing current parameter values
    echo_parts = " ".join(f"{name}=${name}" for name in param_names)
    script_lines.append(f'  echo "Running with {echo_parts}"')

    # Build parameter dict for command formatting (using bash variables)
    bash_params: dict[str, Any] = {
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
    container_runtime: str | None,
    server_image: str | None,
    workload_override: dict[str, Any] | None = None,
) -> list[Path]:
    """Generate Slurm scripts for all workloads."""
    output_files = []

    for dataset, dataset_config in sorted(all_workloads.items()):
        for gpu_model, gpu_workloads in sorted(dataset_config.workloads.items()):
            for num_gpus, workloads in sorted(gpu_workloads.items()):
                for workload in sorted(workloads, key=lambda w: w.model_id):
                    if workload.runtime == "vllm":
                        output_file = generate_vllm_slurm_script(
                            dataset,
                            gpu_model,
                            num_gpus,
                            workload,
                            dataset_config.template,
                            output_dir,
                            slurm_config,
                            container_runtime,
                            server_image,
                            workload_override,
                        )
                    else:
                        output_file = generate_xdit_slurm_script(
                            dataset,
                            gpu_model,
                            num_gpus,
                            workload,
                            dataset_config.template,
                            output_dir,
                            slurm_config,
                        )
                    print(f"Generated {output_file}")
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
    filtered_datasets: dict[str, DatasetConfig] = {}
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

    # Apply sweep override if specified
    if config.override_sweeps:
        override_params = json.loads(config.override_sweeps)
        print(f"Overriding all sweeps with: {override_params}")
        for dataset_config in filtered_datasets.values():
            for gpu_workloads in dataset_config.workloads.values():
                for workloads in gpu_workloads.values():
                    for workload in workloads:
                        workload.sweep_combinations = [override_params]

    # Parse workload override if specified
    workload_override = None
    if config.override_workload:
        workload_override = json.loads(config.override_workload)
        print(f"Overriding all workload parameters with: {workload_override}")

    # Check runtime requirements and warn about mismatches
    has_vllm_workloads = any(
        workload.runtime == "vllm"
        for dataset_config in filtered_datasets.values()
        for gpu_workloads in dataset_config.workloads.values()
        for workloads in gpu_workloads.values()
        for workload in workloads
    )
    has_xdit_workloads = any(
        workload.runtime == "xdit"
        for dataset_config in filtered_datasets.values()
        for gpu_workloads in dataset_config.workloads.values()
        for workloads in gpu_workloads.values()
        for workload in workloads
    )

    # vLLM workloads require container options
    if has_vllm_workloads and (
        config.container_runtime is None or config.server_image is None
    ):
        raise ValueError(
            "vLLM workloads require --container-runtime and --server-image options. "
            "Please specify both, e.g.: --container-runtime docker --server-image vllm/vllm-openai:v0.11.1"
        )

    # xDiT workloads don't use container options
    if has_xdit_workloads and (
        config.container_runtime is not None or config.server_image is not None
    ):
        print(
            "Warning: container_runtime and server_image options are ignored for "
            "xDiT/diffusion workloads (they run as native Python processes)."
        )

    match config.output:
        case Pegasus():
            output_files = generate_pegasus_queues(
                filtered_datasets,
                config.output_dir,
                config.output,
                config.container_runtime,
                config.server_image,
                workload_override,
            )
        case Slurm():
            output_files = generate_slurm_scripts(
                filtered_datasets,
                config.output_dir,
                config.output,
                config.container_runtime,
                config.server_image,
                workload_override,
            )
        case _:
            raise ValueError("Unsupported output configuration")

    print(f"\nGenerated {len(output_files)} output file(s).")

    # Collect unique model IDs for bulk downloading
    unique_model_ids = set()
    for dataset_config in filtered_datasets.values():
        for gpu_workloads in dataset_config.workloads.values():
            for workloads in gpu_workloads.values():
                for workload in workloads:
                    unique_model_ids.add(workload.model_id)

    if unique_model_ids:
        print(f"\n{len(unique_model_ids)} unique model IDs:")
        for model_id in sorted(unique_model_ids):
            print(model_id)


if __name__ == "__main__":
    main(tyro.cli(Generate[Pegasus] | Generate[Slurm]))
