"""Validate benchmark results against expectations.

This script validates benchmark results for various model types. To add support for new model types:
1. Add model type detection in get_model_type()
2. Implement model-specific validators if needed
3. Register validators in validate_result()
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Literal
from multiprocessing import Pool
from collections import defaultdict

import tyro


@dataclass
class Args:
    """CLI arguments.

    Attributes:
        run_dir: Base run directory containing benchmark results.
        verbose: Whether to show details for all runs, not just failed ones.
        workers: Number of parallel workers for validation.
    """

    run_dir: Path = Path("run")
    verbose: bool = False
    workers: int = 4


@dataclass
class RequestResult:
    """A single request result from the benchmark."""

    success: bool
    output_len: int
    error: str | None = None


@dataclass
class PowerTimeline:
    """Power timeline data from results.json."""

    device_instant: dict[str, list[tuple[float, float]]]
    device_average: dict[str, list[tuple[float, float]]]
    memory_average: dict[str, list[tuple[float, float]]]


@dataclass
class PrometheusTimelineEntry:
    """A single entry in the Prometheus timeline."""

    timestamp: float
    metrics: dict


@dataclass
class BenchmarkData:
    """Validated and extracted benchmark data.

    This dataclass holds all data extracted from results.json and prometheus.json,
    validated at load time. All fields that are required for validation are
    non-optional. Optional fields are for data that may legitimately be missing.
    """

    # Identification
    model_id: str
    gpu_model: str
    num_gpus: int
    model_type: Literal["llm", "mllm", "diffusion"]

    # Completion metrics
    completed: int
    num_prompts: int

    # Timing metrics
    duration: float
    steady_state_duration: float

    # Energy metrics
    steady_state_energy: float
    steady_state_energy_per_token: float
    output_throughput: float

    # Configuration
    max_num_seqs: int
    max_output_tokens: int

    # Request results
    request_results: list[RequestResult]

    # Power/temperature timeline
    power_timeline: PowerTimeline
    temperature_timeline: dict[str, list[tuple[float, float]]]

    # Prometheus data
    prom_timeline: list[PrometheusTimelineEntry] = field(default_factory=list)
    prom_steady_start: float = 0.0
    prom_steady_end: float = 0.0

    # Prometheus steady-state stats (may be missing)
    prom_avg_batch_size: float | None = None
    prom_avg_kv_cache: float | None = None  # Already converted to percentage (0-100)
    prom_median_itl: float | None = None  # In seconds

    # Cached computed values for post-validation
    @property
    def output_lengths(self) -> list[int]:
        """Extract output lengths from request results."""
        return [r.output_len for r in self.request_results]

    @property
    def avg_output_length(self) -> float:
        """Calculate average output length."""
        lengths = self.output_lengths
        if not lengths:
            raise ValueError("No output lengths available to calculate average")
        return sum(lengths) / len(lengths)


class DataLoadError(Exception):
    """Raised when required data cannot be loaded or validated."""

    pass


@dataclass
class Expectation:
    """A single validation expectation.

    Attributes:
        name: Name of the expectation being validated.
        passed: Whether the expectation passed.
        message: Short summary message.
        severity: Either "error" or "warning".
        details: Additional context about the failure for debugging.
    """

    name: str
    passed: bool
    message: str
    severity: Literal["error", "warning"] = "error"
    details: str = ""


@dataclass
class ValidationResult:
    """Validation results for a single benchmark run."""

    path: str
    expectations: list[Expectation]
    # Cached data for post-validation (avoids re-reading JSON files)
    data: BenchmarkData | None = None

    @property
    def passed(self) -> bool:
        """Whether all error-level expectations passed."""
        return all(e.passed for e in self.expectations if e.severity == "error")

    @property
    def errors(self) -> list[Expectation]:
        """All failed error-level expectations."""
        return [e for e in self.expectations if not e.passed and e.severity == "error"]

    @property
    def warnings(self) -> list[Expectation]:
        """All failed warning-level expectations."""
        return [
            e for e in self.expectations if not e.passed and e.severity == "warning"
        ]


def get_model_type(result_dir: Path) -> Literal["llm", "mllm", "diffusion"]:
    """Detect model type from directory structure.

    Returns:
        Model type identifier (e.g., "llm", "mllm", "diffusion")
    """
    parts = result_dir.parts
    if "llm" in parts:
        return "llm"
    elif "mllm" in parts:
        return "mllm"
    elif "diffusion" in parts:
        return "diffusion"
    raise ValueError(f"Could not determine model type from path: {result_dir}")


def _require_field(data: dict, field: str, context: str) -> Any:
    """Extract a required field from a dict, raising DataLoadError if missing."""
    if field not in data:
        raise DataLoadError(f"{field} not in {context}")
    value = data[field]
    if value is None:
        raise DataLoadError(f"{field} is null in {context}")
    return value


def _extract_power_timeline(timeline: dict) -> PowerTimeline:
    """Extract power timeline data from results.json timeline."""
    power_data = timeline.get("power")
    if not power_data:
        raise DataLoadError("timeline.power not in results.json")

    def parse_timeline_entries(data: dict) -> dict[str, list[tuple[float, float]]]:
        result = {}
        for gpu_id, entries in data.items():
            parsed = []
            for entry in entries:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    parsed.append((float(entry[0]), float(entry[1])))
            result[gpu_id] = parsed
        return result

    return PowerTimeline(
        device_instant=parse_timeline_entries(power_data["device_instant"]),
        device_average=parse_timeline_entries(power_data["device_average"]),
        memory_average=parse_timeline_entries(power_data["memory_average"]),
    )


def _extract_temperature_timeline(
    timeline: dict,
) -> dict[str, list[tuple[float, float]]]:
    """Extract temperature timeline data from results.json timeline."""
    temp_data = timeline.get("temperature")
    if not temp_data:
        raise DataLoadError("timeline.temperature not in results.json")

    result = {}
    for gpu_id, entries in temp_data.items():
        parsed = []
        for entry in entries:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                parsed.append((float(entry[0]), float(entry[1])))
        result[gpu_id] = parsed
    return result


def load_benchmark_data(result_dir: Path) -> BenchmarkData:
    """Load and validate benchmark data from result directory.

    This function reads results.json and prometheus.json, validates that all
    required fields are present, and returns a structured BenchmarkData object.

    Args:
        result_dir: Path to the result directory containing JSON files.

    Returns:
        BenchmarkData object with all extracted and validated data.

    Raises:
        DataLoadError: If required files are missing or required fields are absent.
        json.JSONDecodeError: If JSON files are malformed.
        FileNotFoundError: If required files don't exist.
    """
    model_type = get_model_type(result_dir)

    # Load results.json
    results_path = result_dir / "results.json"
    with open(results_path) as f:
        results_data = json.load(f)

    # Load prometheus.json
    prom_path = result_dir / "prometheus.json"
    with open(prom_path) as f:
        prom_data = json.load(f)

    # Extract required fields from results.json
    model_id = _require_field(results_data, "model_id", "results.json")
    gpu_model = _require_field(results_data, "gpu_model", "results.json")
    num_gpus = _require_field(results_data, "num_gpus", "results.json")
    completed = _require_field(results_data, "completed", "results.json")
    num_prompts = _require_field(results_data, "num_prompts", "results.json")
    duration = _require_field(results_data, "duration", "results.json")
    steady_state_duration = _require_field(
        results_data, "steady_state_duration", "results.json"
    )
    steady_state_energy = _require_field(
        results_data, "steady_state_energy", "results.json"
    )
    steady_state_energy_per_token = _require_field(
        results_data, "steady_state_energy_per_token", "results.json"
    )
    output_throughput = _require_field(
        results_data, "output_throughput", "results.json"
    )
    max_num_seqs = _require_field(results_data, "max_num_seqs", "results.json")
    max_output_tokens = _require_field(
        results_data, "max_output_tokens", "results.json"
    )

    # Extract request results
    raw_results = _require_field(results_data, "results", "results.json")
    if not isinstance(raw_results, list):
        raise DataLoadError("results field must be a list in results.json")
    if len(raw_results) == 0:
        raise DataLoadError("results array is empty in results.json")

    request_results = []
    for i, r in enumerate(raw_results):
        if "output_len" not in r:
            raise DataLoadError(f"results[{i}].output_len missing in results.json")
        if "success" not in r:
            raise DataLoadError(f"results[{i}].success missing in results.json")
        request_results.append(
            RequestResult(
                success=r["success"],
                output_len=r["output_len"],
                error=r.get("error"),
            )
        )

    # Extract optional power/temperature timeline
    timeline = results_data.get("timeline", {})
    power_timeline = _extract_power_timeline(timeline)
    temperature_timeline = _extract_temperature_timeline(timeline)

    # Extract required fields from prometheus.json
    prom_timeline_raw = _require_field(prom_data, "timeline", "prometheus.json")
    prom_steady_start = _require_field(
        prom_data, "steady_state_start_time", "prometheus.json"
    )
    prom_steady_end = _require_field(
        prom_data, "steady_state_end_time", "prometheus.json"
    )

    # Parse prometheus timeline
    prom_timeline = []
    for entry in prom_timeline_raw:
        if isinstance(entry, dict) and "timestamp" in entry:
            prom_timeline.append(
                PrometheusTimelineEntry(
                    timestamp=entry["timestamp"],
                    metrics=entry.get("metrics", {}),
                )
            )

    # Extract prometheus steady-state stats (optional fields)
    steady_stats = prom_data.get("steady_state_stats", {})
    prom_avg_batch_size = steady_stats.get("vllm:num_requests_running")
    prom_avg_kv_cache_raw = steady_stats.get("vllm:kv_cache_usage_perc")
    prom_avg_kv_cache = (
        prom_avg_kv_cache_raw * 100 if prom_avg_kv_cache_raw is not None else None
    )
    prom_median_itl = steady_stats.get("vllm:inter_token_latency_seconds_p50")

    return BenchmarkData(
        model_id=model_id,
        gpu_model=gpu_model,
        num_gpus=num_gpus,
        model_type=model_type,
        completed=completed,
        num_prompts=num_prompts,
        duration=duration,
        steady_state_duration=steady_state_duration,
        steady_state_energy=steady_state_energy,
        steady_state_energy_per_token=steady_state_energy_per_token,
        output_throughput=output_throughput,
        max_num_seqs=max_num_seqs,
        max_output_tokens=max_output_tokens,
        request_results=request_results,
        power_timeline=power_timeline,
        temperature_timeline=temperature_timeline,
        prom_timeline=prom_timeline,
        prom_steady_start=prom_steady_start,
        prom_steady_end=prom_steady_end,
        prom_avg_batch_size=prom_avg_batch_size,
        prom_avg_kv_cache=prom_avg_kv_cache,
        prom_median_itl=prom_median_itl,
    )


def check_files_present(result_dir: Path) -> Expectation:
    """Check if essential result files are present."""
    results_json = result_dir / "results.json"
    driver_log = result_dir / "driver.log"
    server_log = result_dir / "server.log"
    prometheus_json = result_dir / "prometheus.json"

    missing = []
    if not results_json.exists():
        missing.append("results.json")
    if not driver_log.exists():
        missing.append("driver.log")
    if not server_log.exists():
        missing.append("server.log")
    if not prometheus_json.exists():
        missing.append("prometheus.json")

    if not missing:
        return Expectation(
            "Files Present",
            True,
            "All result files present",
            "error",
        )
    else:
        return Expectation(
            "Files Present",
            False,
            f"Missing: {', '.join(missing)}",
            "error",
        )


def check_completion(data: BenchmarkData) -> Expectation:
    """Check if all requests completed."""
    if data.completed == data.num_prompts:
        return Expectation(
            "Completion",
            True,
            f"{data.completed}/{data.num_prompts} requests completed",
            "error",
            "",
        )
    else:
        details = f"Expected {data.num_prompts} completions but got {data.completed}. Check driver.log for interruptions."
        return Expectation(
            "Completion",
            False,
            f"Only {data.completed}/{data.num_prompts} requests completed",
            "error",
            details,
        )


def check_request_success(data: BenchmarkData) -> Expectation:
    """Check if all requests succeeded."""
    failed = [r for r in data.request_results if not r.success]
    total = len(data.request_results)

    if len(failed) == 0:
        return Expectation(
            "Request Success",
            True,
            f"All {total} requests succeeded",
            "error",
            "",
        )
    else:
        # Collect error messages from failed requests
        error_samples = []
        for r in failed[:3]:  # Show first 3 failures
            error_msg = r.error if r.error else "Unknown error"
            error_samples.append(f"  - {error_msg}")

        details = f"{len(failed)} failed requests. First errors:\n" + "\n".join(
            error_samples
        )
        if len(failed) > 3:
            details += f"\n  ... and {len(failed) - 3} more"

        return Expectation(
            "Request Success",
            False,
            f"{len(failed)}/{total} requests failed",
            "error",
            details,
        )


def check_steady_state_duration(data: BenchmarkData) -> Expectation:
    """Check if steady-state duration is at least 30 seconds."""
    if data.steady_state_duration >= 30:
        return Expectation(
            "Steady State Duration",
            True,
            f"{data.steady_state_duration:.1f}s (>=30s)",
            "error",
            "",
        )
    else:
        details = (
            f"Steady state: {data.steady_state_duration:.1f}s, Total duration: {data.duration:.1f}s. "
            "Increase the number of requests."
        )
        return Expectation(
            "Steady State Duration",
            False,
            f"{data.steady_state_duration:.1f}s (<30s)",
            "error",
            details,
        )


def check_prometheus_collection(data: BenchmarkData) -> Expectation:
    """Check Prometheus collection frequency for both total and steady-state durations."""
    if data.duration == 0:
        return Expectation(
            "Prometheus Collection",
            False,
            "Duration is zero",
            "error",
            "duration must be greater than 0",
        )

    # Check total duration collection rate
    expected_total = int(data.duration)
    actual_total = len(data.prom_timeline)
    ratio_total = actual_total / expected_total if expected_total > 0 else 0

    # Check steady-state collection rate
    # Count timeline entries within steady-state window
    steady_entries = [
        e
        for e in data.prom_timeline
        if data.prom_steady_start <= e.timestamp <= data.prom_steady_end
    ]
    expected_steady = int(data.steady_state_duration)
    actual_steady = len(steady_entries)
    ratio_steady = actual_steady / expected_steady if expected_steady > 0 else 0

    # Check if ratios are acceptable (>=75%)
    issues = []
    if ratio_total < 0.75:
        issues.append(f"total={ratio_total:.2f}")
    if ratio_steady < 0.75:
        issues.append(f"steady={ratio_steady:.2f}")

    if not issues:
        msg = f"Total: {actual_total}/{expected_total} ({ratio_total:.2f}), Steady: {actual_steady}/{expected_steady} ({ratio_steady:.2f})"
        return Expectation(
            "Prometheus Collection",
            True,
            msg,
            "warning",
            "",
        )
    else:
        msg = f"Low collection rate: {', '.join(issues)} (expected >=0.75)"
        details_parts = [
            f"Duration: {data.duration:.1f}s, Timeline entries: {actual_total}",
            f"Total collection rate: {ratio_total:.2%} ({actual_total}/{expected_total})",
            f"Steady-state collection rate: {ratio_steady:.2%} ({actual_steady}/{expected_steady})",
        ]
        details_parts.append(
            "This may indicate CPU contention during multimodal workloads."
        )
        details = "\n".join(details_parts)

        return Expectation(
            "Prometheus Collection",
            False,
            msg,
            "warning",
            details,
        )


def check_metrics_validity(data: BenchmarkData) -> Expectation:
    """Check if key metrics are valid."""
    issues = []

    if data.output_throughput <= 0:
        issues.append(f"throughput={data.output_throughput}")

    if data.steady_state_energy <= 0:
        issues.append(f"energy={data.steady_state_energy}")

    if data.steady_state_energy_per_token <= 0:
        issues.append(f"energy/tok={data.steady_state_energy_per_token}")
    elif (
        data.steady_state_energy_per_token > 100
    ):  # Sanity check: >100J/token is suspicious
        issues.append(f"energy/tok={data.steady_state_energy_per_token:.2f} (>100J)")

    if not issues:
        return Expectation(
            "Metrics Valid",
            True,
            f"throughput={data.output_throughput:.1f}, energy/tok={data.steady_state_energy_per_token:.4f}",
            "error",
            "",
        )
    else:
        details_parts = ["Invalid or suspicious metric values:"]
        if data.output_throughput <= 0:
            details_parts.append(
                f"  - output_throughput: {data.output_throughput} (should be >0)"
            )
        if data.steady_state_energy <= 0:
            details_parts.append(
                f"  - steady_state_energy: {data.steady_state_energy} (should be >0)"
            )
        if data.steady_state_energy_per_token <= 0:
            details_parts.append(
                f"  - steady_state_energy_per_token: {data.steady_state_energy_per_token} (should be >0)"
            )
        elif data.steady_state_energy_per_token > 100:
            details_parts.append(
                f"  - steady_state_energy_per_token: {data.steady_state_energy_per_token:.2f}J (suspiciously high, >100J/token)"
            )
        details_parts.append("Check server.log and driver.log for measurement errors.")
        details = "\n".join(details_parts)

        return Expectation(
            "Metrics Valid",
            False,
            "Invalid metrics: " + ", ".join(issues),
            "error",
            details,
        )


def check_no_crashes(result_dir: Path) -> Expectation:
    """Check driver and server logs for crashes."""
    driver_log = result_dir / "driver.log"
    server_log = result_dir / "server.log"

    crash_indicators = []

    # Check driver log
    if driver_log.exists():
        try:
            with open(driver_log, errors="replace") as f:
                log_content = f.read()

            lines = log_content.split("\n")
            for i, line in enumerate(lines):
                line_lower = line.lower()

                # Look for errors (excluding benign ones)
                if "runtimeerror" in line_lower:
                    crash_indicators.append(f"driver L{i}: RuntimeError")
                elif (
                    "traceback" in line_lower
                    and i + 1 < len(lines)
                    and "error" in lines[i + 1].lower()
                ):
                    crash_indicators.append(f"driver L{i}: Traceback with error")

        except Exception:
            pass  # If we can't read, skip

    # Check server log
    if server_log.exists():
        try:
            with open(server_log, errors="replace") as f:
                log_content = f.read()

            lines = log_content.split("\n")
            for i, line in enumerate(lines):
                line_lower = line.lower()

                # Skip benign messages
                if "no module named" in line_lower and "import error" in line_lower:
                    continue
                if "platform is not available" in line_lower:
                    continue

                # Look for crashes
                if "runtimeerror" in line_lower:
                    crash_indicators.append(f"server L{i}: RuntimeError")
                elif "enginedeaderror" in line_lower:
                    crash_indicators.append(f"server L{i}: EngineDeadError")
                elif "assertion" in line_lower and "failed" in line_lower:
                    crash_indicators.append(f"server L{i}: Assertion failed")
                elif "cuda error" in line_lower or "cuda runtime error" in line_lower:
                    crash_indicators.append(f"server L{i}: CUDA error")

        except Exception:
            pass  # If we can't read, skip

    if not crash_indicators:
        return Expectation(
            "No Crashes",
            True,
            "No crashes detected",
            "error",
            "",
        )
    else:
        # Show first crash indicator
        first_crash = crash_indicators[0]
        if len(crash_indicators) > 1:
            msg = f"{len(crash_indicators)} errors, first: {first_crash}"
        else:
            msg = first_crash

        # Build detailed list of all crash indicators
        details_parts = [f"Found {len(crash_indicators)} crash indicator(s):"]
        for indicator in crash_indicators[:5]:  # Show first 5
            details_parts.append(f"  - {indicator}")
        if len(crash_indicators) > 5:
            details_parts.append(f"  ... and {len(crash_indicators) - 5} more")
        details_parts.append("Check driver.log and server.log for full traceback.")
        details = "\n".join(details_parts)

        return Expectation(
            "No Crashes",
            False,
            msg,
            "error",
            details,
        )


# GPU TDP (Thermal Design Power) values in Watts
GPU_TDP_MAP = {
    "H100": 700,
    "B200": 1000,
}


def get_gpu_tdp(gpu_model: str) -> float | None:
    """Get TDP for a GPU model.

    Args:
        gpu_model: GPU model name from the results.json file
    """
    return GPU_TDP_MAP.get(gpu_model)


def check_power_range(data: BenchmarkData) -> Expectation:
    """Check if power consumption is within reasonable bounds (100W to TDP).

    Args:
        data: Validated benchmark data

    Returns:
        Expectation indicating whether power readings are within acceptable range
    """
    device_instant = data.power_timeline.device_instant
    device_average = data.power_timeline.device_average
    memory_average = data.power_timeline.memory_average

    if not device_instant and not device_average and not memory_average:
        return Expectation(
            "Power Range",
            True,
            "No power readings in timeline",
            "warning",
            "Power timeline exists but contains no readings",
        )

    # Get TDP for this GPU
    tdp = get_gpu_tdp(data.gpu_model)
    if tdp is None:
        return Expectation(
            "Power Range",
            True,
            f"Unknown GPU model: {data.gpu_model}",
            "warning",
            "Cannot validate power without known TDP. Add GPU model to GPU_TDP_MAP.",
        )

    # Check each power type separately
    issues = []
    stats_parts = []

    # Helper function to calculate stats for a power type
    def check_power_type(
        power_dict: dict[str, list[tuple[float, float]]],
        name: str,
        min_threshold: float,
        max_threshold: float,
    ):
        readings = []
        type_issues = []

        for gpu_id, timeline_data in power_dict.items():
            if not timeline_data:
                continue

            for timestamp, power in timeline_data:
                readings.append(power)

                if power < min_threshold:
                    type_issues.append(
                        f"GPU {gpu_id} ({name}): {power:.1f}W < {min_threshold}W"
                    )
                elif power > max_threshold:
                    type_issues.append(
                        f"GPU {gpu_id} ({name}): {power:.1f}W > {max_threshold}W"
                    )

        if readings:
            avg = sum(readings) / len(readings)
            return {
                "name": name,
                "avg": avg,
                "min": min(readings),
                "max": max(readings),
                "count": len(readings),
                "issues": type_issues,
            }
        return None

    # Check device_instant: allow spikes
    instant_power_ceiling_mult = 1.7
    device_instant_stats = None

    if device_instant:
        device_instant_stats = check_power_type(
            device_instant, "device_instant", 70, tdp * instant_power_ceiling_mult
        )
        if device_instant_stats:
            status = "✓" if not device_instant_stats["issues"] else "✗"
            stats_parts.append(
                f"device_instant=avg={device_instant_stats['avg']:.0f}W, "
                f"range=[{device_instant_stats['min']:.0f}, {device_instant_stats['max']:.0f}]W{status}"
            )
            issues.extend(device_instant_stats["issues"])

    # Check device_average: should stay within TDP*1.1 (average should not exceed TDP much)
    device_avg_stats = None
    if device_average:
        device_avg_stats = check_power_type(device_average, "device_avg", 70, tdp * 1.1)
        if device_avg_stats:
            status = "✓" if not device_avg_stats["issues"] else "✗"
            stats_parts.append(
                f"device_avg=avg={device_avg_stats['avg']:.0f}W, "
                f"range=[{device_avg_stats['min']:.0f}, {device_avg_stats['max']:.0f}]W{status}"
            )
            issues.extend(device_avg_stats["issues"])

    # Check memory_average: should be 10W to TDP*0.5
    memory_stats = None
    if memory_average:
        memory_stats = check_power_type(memory_average, "memory", 10, tdp * 0.5)
        if memory_stats:
            status = "✓" if not memory_stats["issues"] else "✗"
            stats_parts.append(
                f"memory=avg={memory_stats['avg']:.0f}W, "
                f"range=[{memory_stats['min']:.0f}, {memory_stats['max']:.0f}]W{status}"
            )
            issues.extend(memory_stats["issues"])

    if not stats_parts:
        return Expectation(
            "Power Range",
            False,
            "No power readings found",
            "error",
            "Power timeline exists but contains no valid readings",
        )

    # Format the summary message
    summary_msg = ", ".join(stats_parts) + f", TDP={tdp}W"

    # Pass if no issues
    if not issues:
        return Expectation(
            "Power Range",
            True,
            summary_msg,
            "warning",
            "",
        )
    else:
        details_parts = [
            f"Power readings for {data.gpu_model} (TDP={tdp}W):",
            f"  Found {len(issues)} out-of-range readings",
            f"  Expected device_instant: [70W, {tdp * instant_power_ceiling_mult}W] (allows spikes)",
            f"  Expected device_avg: [70W, {tdp * 1.1:.0f}W]",
            f"  Expected memory: [0W, {tdp * 0.5:.0f}W]",
        ]

        if issues:
            details_parts.append("\nIssues found:")
            for issue in issues[:10]:
                details_parts.append(f"  - {issue}")
            if len(issues) > 10:
                details_parts.append(f"  ... and {len(issues) - 10} more")

        details_parts.append("\nCheck if power monitoring is configured correctly.")
        details = "\n".join(details_parts)

        return Expectation(
            "Power Range",
            False,
            f"{summary_msg} ({len(issues)} issues)",
            "warning",
            details,
        )


def check_temperature_range(data: BenchmarkData) -> Expectation:
    """Check if GPU temperatures are within reasonable bounds.

    Args:
        data: Validated benchmark data

    Returns:
        Expectation indicating whether temperatures are within acceptable range
    """

    # Reasonable temperature bounds for GPUs under sustained load
    MIN_TEMP = 20  # °C - Below this suggests sensor error
    MAX_TEMP = 95  # °C - Most datacenter GPUs throttle before this

    # Collect all temperature readings across all GPUs
    all_temp_readings = []
    issues = []

    for gpu_id, timeline_data in data.temperature_timeline.items():
        if not timeline_data:
            continue

        for timestamp, temp in timeline_data:
            all_temp_readings.append(temp)

            # Check if temperature is below minimum (sensor error)
            if temp < MIN_TEMP:
                issues.append(
                    f"GPU {gpu_id}: {temp:.1f}°C < {MIN_TEMP}°C (sensor error?)"
                )
            # Check if temperature exceeds maximum (thermal throttling likely)
            elif temp > MAX_TEMP:
                issues.append(
                    f"GPU {gpu_id}: {temp:.1f}°C > {MAX_TEMP}°C (throttling risk)"
                )

    if not all_temp_readings:
        return Expectation(
            "Temperature Range",
            False,
            "No temperature readings found",
            "error",
            "Temperature timeline exists but contains no valid readings",
        )

    # Calculate statistics
    avg_temp = sum(all_temp_readings) / len(all_temp_readings)
    max_temp = max(all_temp_readings)
    min_temp = min(all_temp_readings)

    # Count how many readings are out of range
    out_of_range = sum(1 for t in all_temp_readings if t < MIN_TEMP or t > MAX_TEMP)
    out_of_range_pct = out_of_range / len(all_temp_readings)

    # Fail if >5% of readings are out of range
    if out_of_range_pct >= 0.05:
        details_parts = [
            "Temperature readings outside acceptable range:",
            f"  Average: {avg_temp:.1f}°C",
            f"  Range: [{min_temp:.1f}°C, {max_temp:.1f}°C]",
            f"  Out of range: {out_of_range}/{len(all_temp_readings)} ({out_of_range_pct:.1%})",
            f"  Expected range: [{MIN_TEMP}°C, {MAX_TEMP}°C]",
        ]

        if issues:
            details_parts.append("\nSample issues:")
            for issue in issues[:5]:
                details_parts.append(f"  - {issue}")
            if len(issues) > 5:
                details_parts.append(f"  ... and {len(issues) - 5} more")

        details_parts.append("\nCheck GPU cooling and ensure no thermal throttling.")
        details = "\n".join(details_parts)

        return Expectation(
            "Temperature Range",
            False,
            f"{out_of_range_pct:.1%} readings out of range [{MIN_TEMP}°C, {MAX_TEMP}°C]",
            "error",
            details,
        )
    else:
        return Expectation(
            "Temperature Range",
            True,
            f"avg={avg_temp:.0f}°C, range=[{min_temp:.0f}, {max_temp:.0f}]°C",
            "warning",
            "",
        )


def check_batch_size_saturation(data: BenchmarkData) -> Expectation:
    """Check if batch size is limited by memory saturation.

    If the steady-state average batch size is noticeably smaller than max_num_seqs,
    it suggests that memory has saturated and the batch size cannot increase,
    making the high max_num_seqs setting pointless.

    Args:
        data: Validated benchmark data

    Returns:
        Expectation indicating whether batch size is saturated by memory
    """
    # Fail if we don't have the required metrics
    if data.prom_avg_batch_size is None:
        return Expectation(
            "Batch Size Saturation",
            False,
            "vllm:num_requests_running not in steady_state_stats",
            "error",
            "Batch size metric (vllm:num_requests_running) is required but missing from prometheus.json steady_state_stats",
        )

    # Calculate utilization ratio
    utilization_ratio = (
        data.prom_avg_batch_size / data.max_num_seqs if data.max_num_seqs > 0 else 0
    )

    # Check if batch size is significantly below max_num_seqs
    # Using 80% threshold - if avg batch size is less than 80% of max_num_seqs, flag it
    if utilization_ratio < 0.8:
        msg = f"Avg batch size {data.prom_avg_batch_size:.1f} is {utilization_ratio:.1%} of max_num_seqs={data.max_num_seqs}"

        details_parts = [
            "Steady-state batch size:",
            f"  - Average: {data.prom_avg_batch_size:.1f}",
            f"  - max_num_seqs: {data.max_num_seqs}",
            f"  - average batch size / max_num_seqs: {utilization_ratio:.1%}",
            "",
        ]

        if data.prom_avg_kv_cache is not None:
            details_parts.append("KV cache memory utilization:")
            details_parts.append(f"  - Average: {data.prom_avg_kv_cache:.1f}%")
            details_parts.append("")

            # If memory is high, this explains the saturation
            if data.prom_avg_kv_cache > 85:
                details_parts.append(
                    f"High KV cache usage ({data.prom_avg_kv_cache:.1f}%) indicates memory saturation is limiting batch size. "
                    "It is likely that this max_num_seqs setting can be excluded as it is not achievable."
                )
            else:
                details_parts.append(
                    f"KV cache usage ({data.prom_avg_kv_cache:.1f}%) is not particularly high. "
                    "Batch size may be limited by other factors; this requires further investigation."
                )
        else:
            details_parts.append(
                "KV cache utilization metrics not available. Cannot confirm if memory saturation is the cause."
            )

        details = "\n".join(details_parts)

        return Expectation(
            "Batch Size Saturation",
            False,
            msg,
            "warning",
            details,
        )
    else:
        msg = f"Avg batch size {data.prom_avg_batch_size:.1f} is {utilization_ratio:.1%} of max_num_seqs={data.max_num_seqs}"
        if data.prom_avg_kv_cache is not None:
            msg += f", KV cache {data.prom_avg_kv_cache:.1f}%"
        return Expectation(
            "Batch Size Saturation",
            True,
            msg,
            "warning",
            "",
        )


def check_output_length_saturation(data: BenchmarkData) -> Expectation:
    """Check if output lengths are consistently hitting max limit.

    Args:
        data: Validated benchmark data

    Returns:
        Expectation indicating whether outputs are saturating the max length
    """
    output_lengths = data.output_lengths
    hitting_limit_count = 0
    for output_len in output_lengths:
        if output_len >= data.max_output_tokens:
            hitting_limit_count += 1
            if output_len > data.max_output_tokens:
                print(
                    f"Warning: output_len {output_len} exceeds max_output_tokens {data.max_output_tokens}"
                )

    saturation_rate = hitting_limit_count / len(output_lengths) if output_lengths else 0

    # Calculate statistics
    output_lengths_sorted = sorted(output_lengths)
    n = len(output_lengths_sorted)
    mean = sum(output_lengths) / n if n > 0 else 0
    p50 = output_lengths_sorted[n // 2] if n > 0 else 0
    p75 = output_lengths_sorted[int(n * 0.75)] if n > 0 else 0
    p90 = output_lengths_sorted[int(n * 0.90)] if n > 0 else 0
    p95 = output_lengths_sorted[int(n * 0.95)] if n > 0 else 0
    p99 = output_lengths_sorted[int(n * 0.99)] if n > 0 else 0
    max_len = output_lengths_sorted[-1] if n > 0 else 0

    stats_msg = f"mean={mean:.0f}, p50={p50}, p90={p90}, p99={p99}, max={max_len}"

    if saturation_rate >= 0.3:
        details = (
            f"{hitting_limit_count}/{len(output_lengths)} requests ({saturation_rate:.1%}) "
            f"hit max_output_tokens={data.max_output_tokens}.\n"
            f"Output length statistics: {stats_msg}\n"
            f"Percentiles: p75={p75}, p95={p95}\n"
            f"This suggests outputs may be truncated. Consider increasing max_output_tokens."
        )
        return Expectation(
            "Output Length",
            False,
            f"{saturation_rate:.1%} hit limit; {stats_msg}",
            "warning",
            details,
        )
    else:
        return Expectation(
            "Output Length",
            True,
            f"{saturation_rate:.1%} hit limit; {stats_msg}",
            "warning",
            "",
        )


def validate_result(result_dir: Path) -> ValidationResult:
    """Validate a single benchmark result directory."""
    # Always check if files exist
    files_check = check_files_present(result_dir)

    # If results.json is missing, still check logs for diagnostic info
    results_path = result_dir / "results.json"
    if not results_path.exists():
        expectations = [files_check]
        # Check logs for crash information to help diagnose why results.json wasn't created
        crash_check = check_no_crashes(result_dir)
        expectations.append(crash_check)
        return ValidationResult(str(result_dir), expectations)

    # If files are missing (but results.json exists), return early
    if not files_check.passed:
        return ValidationResult(str(result_dir), [files_check])

    # Try to load and validate benchmark data
    try:
        data = load_benchmark_data(result_dir)
    except DataLoadError as e:
        return ValidationResult(
            str(result_dir),
            [
                files_check,
                Expectation(
                    "Data Validation",
                    False,
                    str(e),
                    "error",
                    f"Required field missing or invalid: {str(e)}",
                ),
            ],
        )
    except json.JSONDecodeError as e:
        return ValidationResult(
            str(result_dir),
            [
                files_check,
                Expectation(
                    "Parse Results",
                    False,
                    f"Error parsing JSON: {e}",
                    "error",
                    f"Failed to parse JSON: {str(e)}",
                ),
            ],
        )
    except FileNotFoundError as e:
        return ValidationResult(
            str(result_dir),
            [
                files_check,
                Expectation(
                    "Files Present",
                    False,
                    f"File not found: {e.filename}",
                    "error",
                    str(e),
                ),
            ],
        )

    expectations = []

    if data.model_type in ["llm", "mllm"]:
        # Run common checks (applicable to all model types)
        expectations.extend(
            [
                files_check,
                check_completion(data),
                check_request_success(data),
                check_output_length_saturation(data),
                check_batch_size_saturation(data),
                check_steady_state_duration(data),
                check_prometheus_collection(data),
                check_metrics_validity(data),
                check_power_range(data),
                check_temperature_range(data),
                check_no_crashes(result_dir),
            ]
        )
    else:
        raise NotImplementedError(
            f"Validation for model type '{data.model_type}' not implemented"
        )

    return ValidationResult(str(result_dir), expectations, data)


def calc_percentiles(values: list[float]) -> dict[str, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "mean": sum(values) / n,
        "p25": sorted_vals[int(n * 0.25)],
        "p50": sorted_vals[int(n * 0.50)],
        "p90": sorted_vals[int(n * 0.90)],
        "p95": sorted_vals[int(n * 0.95)],
        "p99": sorted_vals[int(n * 0.99)],
        "max": sorted_vals[-1],
    }


def print_llm_mllm_statistics(validations: list[ValidationResult]):
    """Print aggregate statistics for LLM/MLLM benchmarks."""
    # Group by task for comparison
    by_task: dict[str, list[dict]] = defaultdict(list)

    for v in validations:
        # Skip if no cached data (validation failed early)
        if v.data is None:
            continue

        data = v.data
        result_path = Path(v.path)

        # Extract task name from path (e.g., run/llm/lm-arena-chat/... or run/mllm/image-chat/...)
        parts = result_path.parts
        task = None
        for i, part in enumerate(parts):
            if part in ["llm", "mllm"] and i + 1 < len(parts):
                task = parts[i + 1]
                break

        # Calculate energy per request = energy_per_token * avg_output_length
        energy_per_req = None
        if data.steady_state_energy_per_token > 0 and data.avg_output_length > 0:
            energy_per_req = data.steady_state_energy_per_token * data.avg_output_length

        # Get median ITL (convert from seconds to milliseconds)
        median_itl = (data.prom_median_itl or 0) * 1000

        # Store for task comparison
        if task and data.steady_state_energy_per_token > 0:
            by_task[task].append(
                {
                    "model_id": data.model_id,
                    "gpu_model": data.gpu_model,
                    "num_gpus": data.num_gpus,
                    "max_num_seqs": data.max_num_seqs,
                    "path": str(result_path),
                    "energy_per_token": data.steady_state_energy_per_token,
                    "steady_state_duration": data.steady_state_duration,
                    "median_itl": median_itl,
                    "energy_per_request": energy_per_req,
                }
            )

    # Print comparison tables by task
    if by_task:
        print("\n" + "=" * 80)
        print("COMPARISON BY TASK")
        print("=" * 80)

        for task in sorted(by_task.keys()):
            runs = by_task[task]

            print(f"\n{task.upper()}:")
            print(
                f"{'Rank':<6} {'Model':<50} {'GPU':<8} {'#GPUs':<7} {'MaxSeqs':<9} {'J/token':<12} {'Steady (s)':<10} {'P50 ITL (ms)':<12} {'J/request':<12}"
            )
            print("-" * 94)

            # Sort by energy per token
            ranked = sorted(runs, key=lambda x: x["energy_per_token"])

            for i, run in enumerate(ranked, 1):
                model = run["model_id"]
                gpu = run["gpu_model"]
                ngpus = run["num_gpus"]
                max_seqs = run["max_num_seqs"]
                e_tok = run["energy_per_token"]
                steady_state_duration = run["steady_state_duration"]
                median_itl = run["median_itl"]
                e_req = run["energy_per_request"]
                max_seqs_str = str(max_seqs) if max_seqs is not None else "N/A"
                e_req_str = f"{e_req:.2f}" if e_req is not None else "N/A"
                print(
                    f"{i:<6} {model:<50} {gpu:<8} {ngpus:<7} {max_seqs_str:<9} {e_tok:<12.4f} {steady_state_duration:<10.1f} {median_itl:<12.1f} {e_req_str:<12}"
                )


def check_max_num_seqs_coverage(validations: list[ValidationResult]) -> None:
    """Check if the largest max_num_seqs in each sweep saturates memory enough.

    For each (model, GPU, num_gpus) combination, finds the result with the largest
    max_num_seqs and checks if its KV cache usage is high enough. If not, adds a
    warning suggesting to test larger max_num_seqs values.

    Args:
        validations: List of validation results to analyze. Modified in-place.
    """
    # Group results by (model_id, gpu_model, num_gpus)
    by_config: dict[tuple[str, str, int], list[ValidationResult]] = defaultdict(list)

    for v in validations:
        # Skip if no cached data (validation failed early)
        if v.data is None:
            continue

        data = v.data
        key = (data.model_id, data.gpu_model, data.num_gpus)
        by_config[key].append(v)

    # For each configuration, check the largest max_num_seqs
    for (model_id, gpu_model, num_gpus), runs in by_config.items():
        # Skip if there's only one run (no sweep)
        if len(runs) <= 1:
            continue

        # Find the run with the largest max_num_seqs
        # Note: we already filtered out None data above, so data is guaranteed to exist
        max_validation = max(runs, key=lambda x: x.data.max_num_seqs if x.data else 0)
        if max_validation.data is None:
            continue
        max_num_seqs = max_validation.data.max_num_seqs
        avg_kv_cache = max_validation.data.prom_avg_kv_cache

        if avg_kv_cache is None:
            continue

        # If KV cache usage is below threshold (e.g., 80%), suggest testing larger values
        # Using 80% as threshold - below this, memory isn't saturated enough
        if avg_kv_cache < 80:
            warning = Expectation(
                "Max Num Seqs Coverage",
                False,
                f"Largest max_num_seqs={max_num_seqs} only reaches {avg_kv_cache:.1f}% KV cache",
                "warning",
                f"Model: {model_id}, GPU: {gpu_model}, Num GPUs: {num_gpus}\n"
                f"The largest max_num_seqs value ({max_num_seqs}) in this sweep only achieves "
                f"{avg_kv_cache:.1f}% KV cache utilization.\n"
                "Consider adding larger max_num_seqs values.",
            )
            max_validation.expectations.append(warning)


def print_aggregate_statistics(validations: list[ValidationResult]):
    """Print aggregate statistics dispatched by model type."""
    # Group validations by model type
    by_model_type: dict[str, list[ValidationResult]] = defaultdict(list)

    for v in validations:
        # Use cached data if available, otherwise try to detect from path
        if v.data is not None:
            by_model_type[v.data.model_type].append(v)
        else:
            try:
                model_type = get_model_type(Path(v.path))
                by_model_type[model_type].append(v)
            except ValueError:
                # Skip if model type cannot be determined
                continue

    print("=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    # Print statistics for each model type
    for model_type in sorted(by_model_type.keys()):
        validations_for_type = by_model_type[model_type]

        if model_type in ["llm", "mllm"]:
            print(f"\n{'LLM/MLLM Benchmarks':.^80}")
            print_llm_mllm_statistics(validations_for_type)

    print()


def main():
    args = tyro.cli(Args)

    # Find all result directories by looking for any of the expected result files
    # This ensures we catch incomplete runs that may be missing results.json
    result_dirs_set = set()
    for pattern in ["results.json", "prometheus.json", "driver.log", "server.log"]:
        for p in args.run_dir.rglob(pattern):
            result_dirs_set.add(p.parent)

    result_dirs = sorted(result_dirs_set)

    if not result_dirs:
        print(f"No result directories found in {args.run_dir}")
        sys.exit(1)

    print(f"Validating {len(result_dirs)} benchmark results...\n")

    # Validate all results in parallel
    with Pool(processes=args.workers) as pool:
        validations = pool.map(validate_result, result_dirs)

    # Perform cross-result validation checks
    check_max_num_seqs_coverage(validations)

    # Categorize results
    passed = [v for v in validations if v.passed]
    failed = [v for v in validations if not v.passed]
    with_warnings = [v for v in validations if v.warnings]

    # Print summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total runs:      {len(validations)}")
    print(f"[PASS] {len(passed)} ({len(passed) / len(validations) * 100:.1f}%)")
    print(f"[FAIL] {len(failed)} ({len(failed) / len(validations) * 100:.1f}%)")
    print(f"[WARN] {len(with_warnings)}")
    print()

    # Print failed runs
    if failed:
        print("=" * 80)
        print(f"FAILED RUNS ({len(failed)})")
        print("=" * 80)
        for validation in failed:
            print(f"\n{validation.path}")
            for exp in validation.errors:
                print(f"  [ERROR] {exp.name}: {exp.message}")
                if exp.details:
                    # Indent details for readability
                    for line in exp.details.split("\n"):
                        print(f"    {line}")
            for exp in validation.warnings:
                print(f"  [WARN] {exp.name}: {exp.message}")
                if exp.details and args.verbose:
                    for line in exp.details.split("\n"):
                        print(f"    {line}")

    # Print runs with warnings (if verbose or no failures)
    if with_warnings and (args.verbose or not failed):
        print("\n" + "=" * 80)
        print(f"RUNS WITH WARNINGS ({len(with_warnings)})")
        print("=" * 80)

        runs_to_show = with_warnings if args.verbose else with_warnings[:10]
        for validation in runs_to_show:
            print(f"\n{validation.path}")
            for exp in validation.warnings:
                print(f"  [WARN] {exp.name}: {exp.message}")
                if exp.details and args.verbose:
                    for line in exp.details.split("\n"):
                        print(f"    {line}")

        if not args.verbose and len(with_warnings) > 10:
            print(f"\n  ... and {len(with_warnings) - 10} more")

    # Print passed runs if verbose
    if args.verbose and passed:
        print("\n" + "=" * 80)
        print(f"PASSED RUNS ({len(passed)})")
        print("=" * 80)
        for validation in passed:
            print(f"\n{validation.path}")
            for exp in validation.expectations:
                if exp.passed:
                    print(f"  [PASS] {exp.name}: {exp.message}")

    # Print aggregate statistics
    print_aggregate_statistics(validations)

    # Exit with error if any runs failed
    if failed:
        print(f"\n{len(failed)} run(s) failed validation")
        sys.exit(1)
    else:
        print(f"\nAll {len(validations)} runs passed validation")
        sys.exit(0)


if __name__ == "__main__":
    main()
