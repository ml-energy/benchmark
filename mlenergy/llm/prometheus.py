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


def calculate_steady_state_avg_stats(
    timeline: list[dict[str, Any]],
    steady_start: float,
    steady_end: float,
    gauge_metric_names: list[str],
) -> dict[str, float]:
    """Calculate average metric values during steady state.

    Args:
        timeline: List of metric snapshots with "timestamp" and "metrics" keys
        steady_start: Steady state start timestamp
        steady_end: Steady state end timestamp
        gauge_metric_names: List of gauge metric names to analyze

    Returns:
        Dict mapping metric names to average values during steady state.
        Example: {"vllm:num_requests_running": 42.3, "vllm:kv_cache_usage_perc": 0.645}
    """
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

    return stats
