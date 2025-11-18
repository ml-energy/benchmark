"""Prometheus metrics collection and analysis for vLLM benchmarks."""

import asyncio
import logging
import re
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class PrometheusCollector:
    """Collects Prometheus metrics from vLLM server."""

    def __init__(self, metrics_url: str, interval: float = 1.0) -> None:
        """Initialize the Prometheus collector.

        Args:
            metrics_url: Metric collection URL of the vLLM server (e.g.,
                "http://127.0.0.1:8000/metrics")
            interval: Collection interval in seconds (default: 1.0)
        """
        self.interval = interval
        self.metrics_url = metrics_url

    async def collect(self, stop_event: asyncio.Event) -> list[dict[str, Any]]:
        """Collect metrics periodically until stop_event is set.

        Args:
            stop_event: Event to signal when to stop collecting

        Returns:
            List of metric snapshots: [{"timestamp": float, "metrics": str}, ...]
        """
        timeline = []
        logger.info(
            f"Starting Prometheus metrics collection from {self.metrics_url} "
            f"(interval: {self.interval}s)"
        )

        # Create session once and reuse it for all requests
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5.0)
        ) as session:
            while not stop_event.is_set():
                try:
                    async with session.get(self.metrics_url) as resp:
                        now = time.time()
                        if resp.status == 200:
                            metrics_text = await resp.text()
                            snapshot = {"timestamp": now, "metrics": metrics_text}
                            timeline.append(snapshot)
                        else:
                            logger.warning(
                                f"Failed to fetch metrics: HTTP {resp.status}"
                            )
                except asyncio.TimeoutError:
                    logger.warning("Timeout fetching Prometheus metrics")
                except Exception as e:
                    logger.warning(f"Error collecting Prometheus metrics: {e}")

                await asyncio.sleep(self.interval)

        logger.info(
            f"Stopped Prometheus collection. Collected {len(timeline)} snapshots"
        )
        return timeline


def parse_gauge(metrics_text: str, metric_name: str) -> dict[str, float]:
    """Parse gauge metric values from Prometheus text format.

    Args:
        metrics_text: Raw Prometheus metrics text
        metric_name: Name of the gauge metric (e.g., "vllm:num_requests_running")

    Returns:
        Dict mapping label combinations to values.
        Example: {"engine=\"0\",model_name=\"...\",pid=\"273\"": 0.653, ...}
        Returns empty dict if metric not found.
    """
    pattern = rf"^{re.escape(metric_name)}\{{([^}}]*)\}}\s+([\d.eE+-]+)"
    matches = re.finditer(pattern, metrics_text, re.MULTILINE)

    results = {}
    for match in matches:
        labels = match.group(1)
        value = float(match.group(2))
        results[labels] = value

    return results


def parse_counter(metrics_text: str, metric_name: str) -> dict[str, float]:
    """Parse counter metric values from Prometheus text format.

    Args:
        metrics_text: Raw Prometheus metrics text
        metric_name: Name of the counter metric (e.g., "vllm:prompt_tokens_total")

    Returns:
        Dict mapping label combinations to values.
        Returns empty dict if metric not found.
    """
    return parse_gauge(metrics_text, metric_name)


def parse_histogram(metrics_text: str, metric_name: str) -> dict[str, Any]:
    """Parse histogram metric from Prometheus text format.

    Args:
        metrics_text: Raw Prometheus metrics text
        metric_name: Name of the histogram metric (e.g., "vllm:time_to_first_token_seconds")

    Returns:
        Dict with "buckets", "sum", and "count" keys.
        Buckets is a dict mapping label+le to count.
        Returns empty dict if metric not found.
    """
    buckets = {}
    sums = {}
    counts = {}

    bucket_pattern = rf"^{re.escape(metric_name)}_bucket\{{([^}}]*)\}}\s+([\d.eE+-]+)"
    sum_pattern = rf"^{re.escape(metric_name)}_sum\{{([^}}]*)\}}\s+([\d.eE+-]+)"
    count_pattern = rf"^{re.escape(metric_name)}_count\{{([^}}]*)\}}\s+([\d.eE+-]+)"

    for match in re.finditer(bucket_pattern, metrics_text, re.MULTILINE):
        labels = match.group(1)
        value = float(match.group(2))
        buckets[labels] = value

    for match in re.finditer(sum_pattern, metrics_text, re.MULTILINE):
        labels = match.group(1)
        value = float(match.group(2))
        sums[labels] = value

    for match in re.finditer(count_pattern, metrics_text, re.MULTILINE):
        labels = match.group(1)
        value = float(match.group(2))
        counts[labels] = value

    return {"buckets": buckets, "sum": sums, "count": counts}


def parse_summary(metrics_text: str, metric_name: str) -> dict[str, Any]:
    """Parse summary metric from Prometheus text format.

    Args:
        metrics_text: Raw Prometheus metrics text
        metric_name: Name of the summary metric

    Returns:
        Dict with "sum" and "count" keys.
        Returns empty dict if metric not found.
    """
    sums = {}
    counts = {}

    sum_pattern = rf"^{re.escape(metric_name)}_sum\{{([^}}]*)\}}\s+([\d.eE+-]+)"
    count_pattern = rf"^{re.escape(metric_name)}_count\{{([^}}]*)\}}\s+([\d.eE+-]+)"

    for match in re.finditer(sum_pattern, metrics_text, re.MULTILINE):
        labels = match.group(1)
        value = float(match.group(2))
        sums[labels] = value

    for match in re.finditer(count_pattern, metrics_text, re.MULTILINE):
        labels = match.group(1)
        value = float(match.group(2))
        counts[labels] = value

    return {"sum": sums, "count": counts}


def _get_gauge_value(
    metrics_text: str, metric_name: str, handle_vllm_bug: bool = False
) -> float | None:
    """Extract a single value from a gauge metric.

    Args:
        metrics_text: Raw Prometheus metrics text
        metric_name: Name of the gauge metric
        handle_vllm_bug: If True, handle vLLM bug where multiple PID entries exist

    Returns:
        Single float value, or None if metric not found
    """
    values = parse_gauge(metrics_text, metric_name)
    if not values:
        return None

    if handle_vllm_bug:
        # vLLM bug: Multiple entries per API server replica with different PIDs.
        nonzero_values = [v for v in values.values() if v > 0]
        if nonzero_values:
            return sum(nonzero_values) / len(nonzero_values)
        return max(values.values())
    else:
        # If there's only one value, return it
        if len(values) == 1:
            return next(iter(values.values()))
        # If multiple values, take the max as a fallback
        return max(values.values())


def _calculate_histogram_percentile(
    histogram_data: dict[str, Any], percentile: float
) -> float | None:
    """Calculate a percentile from Prometheus histogram buckets.

    Args:
        histogram_data: Dict with "buckets", "sum", and "count" keys from parse_histogram
        percentile: Percentile to calculate (0-100, e.g., 50 for median, 95 for P95)

    Returns:
        Percentile value calculated from histogram buckets, or None if cannot be calculated
    """
    buckets = histogram_data.get("buckets", {})
    counts = histogram_data.get("count", {})

    if not buckets or not counts:
        return None

    # Extract bucket upper bounds and cumulative counts
    # Buckets have labels like 'engine="0",le="100.0",model_name="..."'
    bucket_data = []
    for labels, count in buckets.items():
        # Extract the 'le' value (upper bound)
        le_match = re.search(r'le="([^"]+)"', labels)
        if le_match:
            le_str = le_match.group(1)
            if le_str == "+Inf":
                upper_bound = float("inf")
            else:
                upper_bound = float(le_str)
            bucket_data.append((upper_bound, count))

    if not bucket_data:
        return None

    # Sort by upper bound
    bucket_data.sort(key=lambda x: x[0])

    # Get total count
    total_count = bucket_data[-1][1]  # Last bucket has total count
    if total_count == 0:
        return None

    # Find target count for the percentile
    target_count = total_count * (percentile / 100.0)

    # Find the bucket containing the target percentile
    prev_upper = 0.0
    prev_count = 0.0

    for upper_bound, cumulative_count in bucket_data:
        if cumulative_count >= target_count:
            # Target percentile is in this bucket
            if prev_count == cumulative_count:
                # No values in this bucket, use lower bound
                return prev_upper
            # Linear interpolation within bucket
            bucket_width = upper_bound - prev_upper
            count_in_bucket = cumulative_count - prev_count
            fraction = (target_count - prev_count) / count_in_bucket
            result = prev_upper + fraction * bucket_width
            return result

        prev_upper = upper_bound
        prev_count = cumulative_count

    return None


def _calculate_histogram_percentiles(
    histogram_data: dict[str, Any], percentiles: list[float]
) -> dict[str, float]:
    """Calculate multiple percentiles from Prometheus histogram buckets.

    Args:
        histogram_data: Dict with "buckets", "sum", and "count" keys from parse_histogram
        percentiles: List of percentiles to calculate (e.g., [50, 90, 95, 99])

    Returns:
        Dict mapping percentile names to values (e.g., {"p50": 1500.0, "p90": 3000.0})
    """
    results = {}
    for percentile in percentiles:
        value = _calculate_histogram_percentile(histogram_data, percentile)
        if value is not None:
            # Format percentile as p50, p90, p95, p99
            p_name = (
                f"p{int(percentile)}"
                if percentile == int(percentile)
                else f"p{percentile}"
            )
            results[p_name] = value
    return results


def calculate_steady_state_stats(
    timeline: list[dict[str, Any]],
    steady_start: float,
    steady_end: float,
    gauge_metric_names: list[str],
    histogram_metric_names: list[str] | None = None,
    histogram_percentiles: list[float] | None = None,
) -> dict[str, float]:
    """Calculate average metric values during steady state.

    Args:
        timeline: List of metric snapshots with "timestamp" and "metrics" keys
        steady_start: Steady state start timestamp
        steady_end: Steady state end timestamp
        gauge_metric_names: List of gauge metric names to analyze
        histogram_metric_names: List of histogram metric names to calculate percentiles for
        histogram_percentiles: Percentiles to calculate for histograms (default: [50, 90, 95, 99])

    Returns:
        Dict mapping metric names to values during steady state.
        For gauges: average values.
        For histograms: percentile values with "_p50", "_p90", etc. suffixes.
        Example: {
            "vllm:num_requests_running": 42.3,
            "vllm:request_prompt_tokens_p50": 1500.0,
            "vllm:request_prompt_tokens_p90": 2800.0,
            "vllm:request_prompt_tokens_p95": 3200.0,
            "vllm:request_prompt_tokens_p99": 4100.0,
        }
    """
    if histogram_percentiles is None:
        histogram_percentiles = [50, 90, 95, 99]
    # Filter timeline to steady state window
    steady_snapshots = [
        snapshot
        for snapshot in timeline
        if steady_start <= snapshot["timestamp"] <= steady_end
    ]

    if not steady_snapshots:
        logger.warning(
            f"No Prometheus snapshots found in steady state window "
            f"({steady_start} - {steady_end})"
        )
        return {}

    logger.info(
        f"Analyzing {len(steady_snapshots)} Prometheus snapshots "
        f"during steady state window"
    )

    stats = {}

    # Process gauge metrics
    for gauge_metric_name in gauge_metric_names:
        values = []
        # Special handling for vLLM KV cache bug
        handle_bug = gauge_metric_name == "vllm:kv_cache_usage_perc"

        for snapshot in steady_snapshots:
            value = _get_gauge_value(snapshot["metrics"], gauge_metric_name, handle_bug)
            if value is not None:
                values.append(value)

        if values:
            avg_value = sum(values) / len(values)
            stats[gauge_metric_name] = avg_value
            logger.info(
                f"{gauge_metric_name}: {avg_value:.3f} (averaged over {len(values)} snapshots)"
            )
        else:
            logger.warning(f"No values found for metric: {gauge_metric_name}")

    # Process histogram metrics (calculate percentiles from final snapshot)
    if histogram_metric_names:
        # Use the last snapshot in steady state for histogram metrics
        # (histograms are cumulative, so we want the final state)
        if steady_snapshots:
            final_snapshot = steady_snapshots[-1]
            for histogram_metric_name in histogram_metric_names:
                histogram_data = parse_histogram(
                    final_snapshot["metrics"], histogram_metric_name
                )
                percentile_values = _calculate_histogram_percentiles(
                    histogram_data, histogram_percentiles
                )
                if percentile_values:
                    # Store each percentile with suffix (e.g., "_p50", "_p90")
                    for p_name, value in percentile_values.items():
                        stats[f"{histogram_metric_name}_{p_name}"] = value

                    # Log summary
                    p_str = ", ".join(
                        f"{p_name}={value:.3f}"
                        for p_name, value in percentile_values.items()
                    )
                    logger.info(f"{histogram_metric_name}: {p_str}")
                else:
                    logger.warning(
                        f"Could not calculate percentiles for histogram: {histogram_metric_name}"
                    )

    return stats
