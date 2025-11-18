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

import tyro


@dataclass
class Args:
    """CLI arguments.

    Attributes:
        base_dir: Base directory containing benchmark results.
        verbose: Whether to show details for all runs, not just failed ones.
        workers: Number of parallel workers for validation.
    """

    base_dir: Path = Path("run")
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
    completed = data.get("completed", 0)
    num_prompts = data.get("num_prompts", 0)

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
    steady_duration = data.get("steady_state_duration", 0)

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

    timeline = prom_data.get("timeline", [])
    duration = data.get("duration", 0)
    steady_duration = data.get("steady_state_duration", 0)

    steady_start = prom_data.get("steady_state_start_time")
    steady_end = prom_data.get("steady_state_end_time")

    if duration == 0:
        return Expectation(
            "Prometheus Collection",
            False,
            "Duration is zero",
            "error",
        )

    # Check total duration collection rate
    expected_total = int(duration)
    actual_total = len(timeline)
    ratio_total = actual_total / expected_total if expected_total > 0 else 0

    # Check steady-state collection rate if we have timing info
    ratio_steady = None
    actual_steady = 0
    expected_steady = 0
    if steady_start is not None and steady_end is not None and timeline:
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
    if ratio_steady is not None and ratio_steady < 0.75:
        issues.append(f"steady={ratio_steady:.2f}")

    if not issues:
        msg = f"Total: {actual_total}/{expected_total} ({ratio_total:.2f})"
        if ratio_steady is not None:
            msg += f", Steady: {actual_steady}/{expected_steady} ({ratio_steady:.2f})"
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
        ]
        if ratio_steady is not None:
            details_parts.append(
                f"Steady-state collection rate: {ratio_steady:.2%} ({actual_steady}/{expected_steady})"
            )
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

    # Skip check if no max limit or no results
    if max_output_tokens is None or not results:
        return Expectation(
            "Output Length",
            True,
            "No max_output_tokens limit set",
            "warning",
            "",
        )

    if not isinstance(max_output_tokens, int):
        return Expectation(
            "Output Length",
            True,
            f"max_output_tokens is not an integer: {max_output_tokens}",
            "warning",
            "",
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
                check_steady_state_duration(result_dir, data),
                check_prometheus_collection(result_dir, data),
                check_metrics_validity(result_dir, data),
                check_no_crashes(result_dir, data),
            ]
        )
    else:
        raise NotImplementedError(
            f"Validation for model type '{model_type}' not implemented"
        )

    return ValidationResult(str(result_dir), expectations)


def main():
    args = tyro.cli(Args)

    # Find all result directories by looking for any of the expected result files
    # This ensures we catch incomplete runs that may be missing results.json
    result_dirs_set = set()
    for pattern in ["results.json", "prometheus.json", "driver.log", "server.log"]:
        for p in args.base_dir.rglob(pattern):
            result_dirs_set.add(p.parent)

    result_dirs = sorted(result_dirs_set)

    if not result_dirs:
        print(f"No result directories found in {args.base_dir}")
        sys.exit(1)

    print(f"Validating {len(result_dirs)} benchmark results...\n")

    # Validate all results in parallel
    with Pool(processes=args.workers) as pool:
        validations = pool.map(validate_result, result_dirs)

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

    # Exit with error if any runs failed
    if failed:
        print(f"\n{len(failed)} run(s) failed validation")
        sys.exit(1)
    else:
        print(f"\nAll {len(validations)} runs passed validation")
        sys.exit(0)


if __name__ == "__main__":
    main()
