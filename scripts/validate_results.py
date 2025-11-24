"""Validate benchmark results against expectations.

This script validates benchmark results for various model types. To add support for new model types:
1. Add model type detection in get_model_type()
2. Implement model-specific validators if needed
3. Register validators in validate_result()
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
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


def check_completion(result_dir: Path, data: dict) -> Expectation:
    """Check if all requests completed."""
    completed = data.get("completed")
    num_prompts = data.get("num_prompts")

    # Fail if required fields are missing
    if completed is None:
        return Expectation(
            "Completion",
            False,
            "completed not in results.json",
            "error",
            "completed field is required in results.json",
        )

    if num_prompts is None:
        return Expectation(
            "Completion",
            False,
            "num_prompts not in results.json",
            "error",
            "num_prompts field is required in results.json",
        )

    if completed == num_prompts:
        return Expectation(
            "Completion",
            True,
            f"{completed}/{num_prompts} requests completed",
            "error",
            "",
        )
    else:
        details = f"Expected {num_prompts} completions but got {completed}. Check driver.log for interruptions."
        return Expectation(
            "Completion",
            False,
            f"Only {completed}/{num_prompts} requests completed",
            "error",
            details,
        )


def check_request_success(result_dir: Path, data: dict) -> Expectation:
    """Check if all requests succeeded."""
    results = data.get("results", [])
    failed = [r for r in results if not r.get("success", False)]

    if not results:
        return Expectation(
            "Request Success",
            False,
            "No request results found",
            "error",
            "results.json contains empty or missing 'results' array",
        )

    if len(failed) == 0:
        return Expectation(
            "Request Success",
            True,
            f"All {len(results)} requests succeeded",
            "error",
            "",
        )
    else:
        # Collect error messages from failed requests
        error_samples = []
        for r in failed[:3]:  # Show first 3 failures
            error_msg = r.get("error", "Unknown error")
            error_samples.append(f"  - {error_msg}")

        details = f"{len(failed)} failed requests. First errors:\n" + "\n".join(
            error_samples
        )
        if len(failed) > 3:
            details += f"\n  ... and {len(failed) - 3} more"

        return Expectation(
            "Request Success",
            False,
            f"{len(failed)}/{len(results)} requests failed",
            "error",
            details,
        )


def check_steady_state_duration(result_dir: Path, data: dict) -> Expectation:
    """Check if steady-state duration is at least 30 seconds."""
    steady_duration = data.get("steady_state_duration")

    # Fail if required field is missing
    if steady_duration is None:
        return Expectation(
            "Steady State Duration",
            False,
            "steady_state_duration not in results.json",
            "error",
            "steady_state_duration field is required in results.json",
        )

    if steady_duration >= 30:
        return Expectation(
            "Steady State Duration",
            True,
            f"{steady_duration:.1f}s (>=30s)",
            "error",
            "",
        )
    else:
        total_duration = data.get("duration", 0)
        details = (
            f"Steady state: {steady_duration:.1f}s, Total duration: {total_duration:.1f}s. "
            "Increase the number of requests."
        )
        return Expectation(
            "Steady State Duration",
            False,
            f"{steady_duration:.1f}s (<30s)",
            "error",
            details,
        )


def check_prometheus_collection(result_dir: Path, data: dict) -> Expectation:
    """Check Prometheus collection frequency for both total and steady-state durations."""
    prom_path = result_dir / "prometheus.json"

    if not prom_path.exists():
        return Expectation(
            "Prometheus Collection",
            False,
            "prometheus.json missing",
            "error",
        )

    try:
        with open(prom_path) as f:
            prom_data = json.load(f)
    except Exception as e:
        return Expectation(
            "Prometheus Collection",
            False,
            f"Error reading prometheus.json: {e}",
            "error",
        )

    timeline = prom_data.get("timeline")
    duration = data.get("duration")
    steady_duration = data.get("steady_state_duration")

    steady_start = prom_data.get("steady_state_start_time")
    steady_end = prom_data.get("steady_state_end_time")

    # Fail if required fields are missing
    if duration is None:
        return Expectation(
            "Prometheus Collection",
            False,
            "duration not in results.json",
            "error",
            "duration field is required for Prometheus collection validation",
        )

    if steady_duration is None:
        return Expectation(
            "Prometheus Collection",
            False,
            "steady_state_duration not in results.json",
            "error",
            "steady_state_duration field is required for Prometheus collection validation",
        )

    if timeline is None:
        return Expectation(
            "Prometheus Collection",
            False,
            "timeline not in prometheus.json",
            "error",
            "timeline field is required in prometheus.json",
        )

    if steady_start is None:
        return Expectation(
            "Prometheus Collection",
            False,
            "steady_state_start_time not in prometheus.json",
            "error",
            "steady_state_start_time field is required in prometheus.json",
        )

    if steady_end is None:
        return Expectation(
            "Prometheus Collection",
            False,
            "steady_state_end_time not in prometheus.json",
            "error",
            "steady_state_end_time field is required in prometheus.json",
        )

    if duration == 0:
        return Expectation(
            "Prometheus Collection",
            False,
            "Duration is zero",
            "error",
            "duration must be greater than 0",
        )

    # Check total duration collection rate
    expected_total = int(duration)
    actual_total = len(timeline)
    ratio_total = actual_total / expected_total if expected_total > 0 else 0

    # Check steady-state collection rate
    # Count timeline entries within steady-state window
    steady_entries = [
        e for e in timeline if steady_start <= e.get("timestamp", 0) <= steady_end
    ]
    expected_steady = int(steady_duration)
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
            f"Duration: {duration:.1f}s, Timeline entries: {actual_total}",
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


def check_metrics_validity(result_dir: Path, data: dict) -> Expectation:
    """Check if key metrics are valid."""
    issues = []

    throughput = data.get("output_throughput")
    if throughput is None or throughput <= 0:
        issues.append(f"throughput={throughput}")

    energy = data.get("steady_state_energy")
    if energy is None or energy <= 0:
        issues.append(f"energy={energy}")

    energy_per_tok = data.get("steady_state_energy_per_token")
    if energy_per_tok is None or energy_per_tok <= 0:
        issues.append(f"energy/tok={energy_per_tok}")
    elif energy_per_tok > 100:  # Sanity check: >100J/token is suspicious
        issues.append(f"energy/tok={energy_per_tok:.2f} (>100J)")

    if not issues:
        return Expectation(
            "Metrics Valid",
            True,
            f"throughput={throughput:.1f}, energy/tok={energy_per_tok:.4f}",
            "error",
            "",
        )
    else:
        details_parts = ["Invalid or suspicious metric values:"]
        if throughput is not None and throughput <= 0:
            details_parts.append(f"  - output_throughput: {throughput} (should be >0)")
        if energy is not None and energy <= 0:
            details_parts.append(f"  - steady_state_energy: {energy} (should be >0)")
        if energy_per_tok is not None and energy_per_tok <= 0:
            details_parts.append(
                f"  - steady_state_energy_per_token: {energy_per_tok} (should be >0)"
            )
        elif energy_per_tok is not None and energy_per_tok > 100:
            details_parts.append(
                f"  - steady_state_energy_per_token: {energy_per_tok:.2f}J (suspiciously high, >100J/token)"
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


def check_no_crashes(result_dir: Path, data: dict) -> Expectation:
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
    "H200": 700,
    "B200": 1000,
}


def get_gpu_tdp(gpu_model: str) -> float | None:
    """Get TDP for a GPU model.

    Args:
        gpu_model: GPU model name from the results.json file
    """
    return GPU_TDP_MAP.get(gpu_model)


def check_power_range(result_dir: Path, data: dict) -> Expectation:
    """Check if power consumption is within reasonable bounds (100W to TDP).

    Args:
        result_dir: Path to result directory
        data: Parsed results.json data

    Returns:
        Expectation indicating whether power readings are within acceptable range
    """
    gpu_model = data.get("gpu_model", "unknown")
    timeline = data.get("timeline", {})
    power_data = timeline.get("power", {})

    # Check all power fields: device_instant, device_average, memory_average
    device_instant = power_data.get("device_instant", {})
    device_average = power_data.get("device_average", {})
    memory_average = power_data.get("memory_average", {})

    if not device_instant and not device_average and not memory_average:
        return Expectation(
            "Power Range",
            True,
            "No power data to validate",
            "warning",
            "Power timeline not found in results.json",
        )

    # Get TDP for this GPU
    tdp = get_gpu_tdp(gpu_model)
    if tdp is None:
        return Expectation(
            "Power Range",
            True,
            f"Unknown GPU model: {gpu_model}",
            "warning",
            "Cannot validate power without known TDP. Add GPU model to GPU_TDP_MAP.",
        )

    # Check each power type separately
    issues = []
    stats_parts = []

    # Helper function to calculate stats for a power type
    def check_power_type(power_dict, name, min_threshold, max_threshold):
        readings = []
        type_issues = []

        for gpu_id, timeline_data in power_dict.items():
            if not timeline_data:
                continue

            for entry in timeline_data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    power = entry[1]
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
    instant_power_ceiling_mult = 1.5
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
            f"Power readings for {gpu_model} (TDP={tdp}W):",
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


def check_temperature_range(result_dir: Path, data: dict) -> Expectation:
    """Check if GPU temperatures are within reasonable bounds.

    Args:
        result_dir: Path to result directory
        data: Parsed results.json data

    Returns:
        Expectation indicating whether temperatures are within acceptable range
    """
    timeline = data.get("timeline", {})
    temperature_timeline = timeline.get("temperature", {})

    if not temperature_timeline:
        return Expectation(
            "Temperature Range",
            True,
            "No temperature data to validate",
            "warning",
            "Temperature timeline not found in results.json",
        )

    # Reasonable temperature bounds for GPUs under sustained load
    MIN_TEMP = 20  # °C - Below this suggests sensor error
    MAX_TEMP = 95  # °C - Most datacenter GPUs throttle before this

    # Collect all temperature readings across all GPUs
    all_temp_readings = []
    issues = []

    for gpu_id, timeline_data in temperature_timeline.items():
        if not timeline_data:
            continue

        for entry in timeline_data:
            # Timeline format: [[timestamp, temperature], ...]
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                temp = entry[1]
            else:
                continue

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


def check_batch_size_saturation(result_dir: Path, data: dict) -> Expectation:
    """Check if batch size is limited by memory saturation.

    If the steady-state average batch size is noticeably smaller than max_num_seqs,
    it suggests that memory has saturated and the batch size cannot increase,
    making the high max_num_seqs setting pointless.

    Args:
        result_dir: Path to result directory
        data: Parsed results.json data

    Returns:
        Expectation indicating whether batch size is saturated by memory
    """
    max_num_seqs = data.get("max_num_seqs")

    # Fail if max_num_seqs is not set (it should be present for benchmarks)
    if max_num_seqs is None:
        return Expectation(
            "Batch Size Saturation",
            False,
            "max_num_seqs not in results.json",
            "error",
            "max_num_seqs is required for batch size validation",
        )

    prom_path = result_dir / "prometheus.json"
    if not prom_path.exists():
        return Expectation(
            "Batch Size Saturation",
            False,
            "prometheus.json missing",
            "error",
            "prometheus.json is required for batch size validation",
        )

    try:
        with open(prom_path) as f:
            prom_data = json.load(f)
    except Exception as e:
        return Expectation(
            "Batch Size Saturation",
            False,
            f"Error reading prometheus.json: {e}",
            "error",
            f"Failed to parse prometheus.json: {str(e)}",
        )

    steady_stats = prom_data.get("steady_state_stats", {})

    # Get average batch size (num_requests_running) - stored as single mean value
    avg_batch_size = steady_stats.get("vllm:num_requests_running")

    # Get memory utilization - stored as single mean value (as fraction 0-1, convert to percentage)
    avg_kv_cache_raw = steady_stats.get("vllm:kv_cache_usage_perc")
    avg_kv_cache = avg_kv_cache_raw * 100 if avg_kv_cache_raw is not None else None

    # Fail if we don't have the required metrics
    if avg_batch_size is None:
        return Expectation(
            "Batch Size Saturation",
            False,
            "vllm:num_requests_running not in steady_state_stats",
            "error",
            "Batch size metric (vllm:num_requests_running) is required but missing from prometheus.json steady_state_stats",
        )

    # Calculate utilization ratio
    utilization_ratio = avg_batch_size / max_num_seqs if max_num_seqs > 0 else 0

    # Check if batch size is significantly below max_num_seqs
    # Using 60% threshold - if avg batch size is less than 80% of max_num_seqs, flag it
    if utilization_ratio < 0.8:
        msg = f"Avg batch size {avg_batch_size:.1f} is {utilization_ratio:.1%} of max_num_seqs={max_num_seqs}"

        details_parts = [
            "Steady-state batch size:",
            f"  - Average: {avg_batch_size:.1f}",
            f"  - max_num_seqs: {max_num_seqs}",
            f"  - average batch size / max_num_seqs: {utilization_ratio:.1%}",
            "",
        ]

        if avg_kv_cache is not None:
            # Convert to percentage if needed (value is already 0-100)
            details_parts.append("KV cache memory utilization:")
            details_parts.append(f"  - Average: {avg_kv_cache:.1f}%")
            details_parts.append("")

            # If memory is high, this explains the saturation
            if avg_kv_cache > 85:
                details_parts.append(
                    f"High KV cache usage ({avg_kv_cache:.1f}%) indicates memory saturation is limiting batch size. "
                    "It is likely that this max_num_seqs setting can be excluded as it is not achievable."
                )
            else:
                details_parts.append(
                    f"KV cache usage ({avg_kv_cache:.1f}%) is not particularly high. "
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
        msg = f"Avg batch size {avg_batch_size:.1f} is {utilization_ratio:.1%} of max_num_seqs={max_num_seqs}"
        if avg_kv_cache is not None:
            msg += f", KV cache {avg_kv_cache:.1f}%"
        return Expectation(
            "Batch Size Saturation",
            True,
            msg,
            "warning",
            "",
        )


def check_output_length_saturation(result_dir: Path, data: dict) -> Expectation:
    """Check if output lengths are consistently hitting max limit.

    Args:
        result_dir: Path to result directory
        data: Parsed results.json data

    Returns:
        Expectation indicating whether outputs are saturating the max length
    """
    max_output_tokens = data.get("max_output_tokens")
    results = data.get("results", [])

    # Fail if max_output_tokens is not set (it should be present for benchmarks)
    if max_output_tokens is None:
        return Expectation(
            "Output Length",
            False,
            "max_output_tokens not in results.json",
            "error",
            "max_output_tokens is required for output length validation",
        )

    # Fail if results are missing
    if not results:
        return Expectation(
            "Output Length",
            False,
            "No results found in results.json",
            "error",
            "Results array is required for output length validation",
        )

    # Fail if max_output_tokens is not an integer
    if not isinstance(max_output_tokens, int):
        return Expectation(
            "Output Length",
            False,
            f"max_output_tokens is not an integer: {max_output_tokens}",
            "error",
            f"max_output_tokens must be an integer, got {type(max_output_tokens).__name__}",
        )

    hitting_limit = []
    output_lengths = []
    for r in results:
        output_len = r["output_len"]
        output_lengths.append(output_len)
        if output_len == max_output_tokens:
            hitting_limit.append(r)
        elif output_len > max_output_tokens:
            print(
                f"Warning: output_len {output_len} exceeds max_output_tokens {max_output_tokens}"
            )
            hitting_limit.append(r)

    saturation_rate = len(hitting_limit) / len(results) if results else 0

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
            f"{len(hitting_limit)}/{len(results)} requests ({saturation_rate:.1%}) "
            f"hit max_output_tokens={max_output_tokens}.\n"
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


def get_model_type(result_dir: Path) -> Literal["llm", "mllm", "diffusion"]:
    """Detect model type from directory structure.

    Returns:
        Model type identifier (e.g., "llm", "mllm", "diffusion")
    """
    # Check if path contains llm or mllm
    parts = result_dir.parts
    if "llm" in parts:
        return "llm"
    elif "mllm" in parts:
        return "mllm"
    elif "diffusion" in parts:
        return "diffusion"
    raise ValueError(f"Could not determine model type from path: {result_dir}")


def validate_result(result_dir: Path) -> ValidationResult:
    """Validate a single benchmark result directory."""
    # Detect model type for potential model-specific validation
    model_type = get_model_type(result_dir)

    # Always check if files exist
    files_check = check_files_present(result_dir)

    # If results.json is missing, still check logs for diagnostic info
    results_path = result_dir / "results.json"
    if not results_path.exists():
        expectations = [files_check]
        # Check logs for crash information to help diagnose why results.json wasn't created
        crash_check = check_no_crashes(result_dir, {})
        expectations.append(crash_check)
        return ValidationResult(str(result_dir), expectations)

    # If files are missing (but results.json exists), return early
    if not files_check.passed:
        return ValidationResult(str(result_dir), [files_check])

    # Try to parse results.json
    try:
        with open(results_path) as f:
            data = json.load(f)
    except Exception as e:
        return ValidationResult(
            str(result_dir),
            [
                files_check,
                Expectation(
                    "Parse Results",
                    False,
                    f"Error reading results.json: {e}",
                    "error",
                    f"Failed to parse JSON: {str(e)}",
                ),
            ],
        )

    expectations = []

    if model_type in ["llm", "mllm"]:
        # Run common checks (applicable to all model types)
        expectations.extend(
            [
                files_check,
                check_completion(result_dir, data),
                check_request_success(result_dir, data),
                check_output_length_saturation(result_dir, data),
                check_batch_size_saturation(result_dir, data),
                check_steady_state_duration(result_dir, data),
                check_prometheus_collection(result_dir, data),
                check_metrics_validity(result_dir, data),
                check_power_range(result_dir, data),
                check_temperature_range(result_dir, data),
                check_no_crashes(result_dir, data),
            ]
        )
    else:
        raise NotImplementedError(
            f"Validation for model type '{model_type}' not implemented"
        )

    return ValidationResult(str(result_dir), expectations)


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
        result_path = Path(v.path)
        results_json = result_path / "results.json"
        prometheus_json = result_path / "prometheus.json"

        try:
            with open(results_json) as f:
                data = json.load(f)
        except Exception:
            continue

        try:
            with open(prometheus_json) as f:
                prometheus_stats = json.load(f)["steady_state_stats"]
        except Exception:
            prometheus_stats = None

        # Extract task name from path (e.g., run/llm/lm-arena-chat/... or run/mllm/image-chat/...)
        parts = result_path.parts
        task = None
        for i, part in enumerate(parts):
            if part in ["llm", "mllm"] and i + 1 < len(parts):
                task = parts[i + 1]
                break

        # Collect metrics
        energy_tok = data.get("steady_state_energy_per_token")
        steady_state_duration = data.get("steady_state_duration")
        model_id = data.get("model_id", "unknown")
        gpu_model = data.get("gpu_model", "unknown")
        num_gpus = data.get("num_gpus", 0)
        median_itl = (
            prometheus_stats.get("vllm:inter_token_latency_seconds_p50", 0)
            if prometheus_stats is not None
            else 0
        )
        results = data.get("results", [])

        # Calculate energy per request = energy_per_token * avg_output_length
        energy_per_req = None
        if energy_tok is not None and energy_tok > 0 and results:
            output_lengths = [r["output_len"] for r in results]
            avg_output_len = (
                sum(output_lengths) / len(output_lengths) if output_lengths else 0
            )
            if avg_output_len > 0:
                energy_per_req = energy_tok * avg_output_len

        # Store for task comparison
        if task and energy_tok is not None and energy_tok > 0:
            max_num_seqs = data.get("max_num_seqs")
            by_task[task].append(
                {
                    "model_id": model_id,
                    "gpu_model": gpu_model,
                    "num_gpus": num_gpus,
                    "max_num_seqs": max_num_seqs,
                    "path": str(result_path),
                    "energy_per_token": energy_tok,
                    "steady_state_duration": steady_state_duration,
                    "median_itl": median_itl * 1000,
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
    by_config: dict[tuple[str, str, int], list[tuple[ValidationResult, dict]]] = (
        defaultdict(list)
    )

    for v in validations:
        result_path = Path(v.path)
        results_json = result_path / "results.json"

        # Skip if results.json doesn't exist or failed validation
        if not results_json.exists():
            continue

        try:
            with open(results_json) as f:
                data = json.load(f)
        except Exception:
            continue

        model_id = data.get("model_id")
        gpu_model = data.get("gpu_model")
        num_gpus = data.get("num_gpus")
        max_num_seqs = data.get("max_num_seqs")

        # Skip if required fields are missing
        if not all([model_id, gpu_model, num_gpus is not None, max_num_seqs]):
            continue

        key = (model_id, gpu_model, num_gpus)
        by_config[key].append((v, data))

    # For each configuration, check the largest max_num_seqs
    for (model_id, gpu_model, num_gpus), runs in by_config.items():
        # Skip if there's only one run (no sweep)
        if len(runs) <= 1:
            continue

        # Find the run with the largest max_num_seqs
        max_run = max(runs, key=lambda x: x[1].get("max_num_seqs", 0))
        max_validation, max_data = max_run
        max_num_seqs = max_data.get("max_num_seqs")

        # Get KV cache usage from prometheus.json
        prom_path = Path(max_validation.path) / "prometheus.json"
        if not prom_path.exists():
            continue

        try:
            with open(prom_path) as f:
                prom_data = json.load(f)
        except Exception:
            continue

        steady_stats = prom_data.get("steady_state_stats", {})
        avg_kv_cache_raw = steady_stats.get("vllm:kv_cache_usage_perc")

        if avg_kv_cache_raw is None:
            continue

        avg_kv_cache = avg_kv_cache_raw * 100  # Convert to percentage

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
