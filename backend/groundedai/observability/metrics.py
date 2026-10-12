"""In-process operational metrics (counters and latency samples).

Values live in memory and reset when the process restarts; question outcomes
are also persisted as ``query_events`` rows for the dashboard. Exposed as JSON
and in the Prometheus text format so any scraper can collect them; no paid
monitoring service is required.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping

LabelSet = tuple[tuple[str, str], ...]
MAX_SAMPLES = 2048


def _labels(labels: Mapping[str, object]) -> LabelSet:
    return tuple(sorted((key, str(value)) for key, value in labels.items()))


def percentile(values: Iterable[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 100]); ``None`` for no data."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    data = list(values)
    if not data:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(data),
        "mean": round(sum(data) / len(data), 3),
        "p50": round(percentile(data, 50) or 0.0, 3),
        "p95": round(percentile(data, 95) or 0.0, 3),
        "max": round(max(data), 3),
    }


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, LabelSet], int] = {}
        self._samples: dict[tuple[str, LabelSet], deque[float]] = {}
        self.started_at = time.time()

    def inc(self, name: str, amount: int = 1, **labels: object) -> None:
        key = (name, _labels(labels))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    def observe(self, name: str, value_ms: float, **labels: object) -> None:
        key = (name, _labels(labels))
        with self._lock:
            samples = self._samples.get(key)
            if samples is None:
                samples = self._samples[key] = deque(maxlen=MAX_SAMPLES)
            samples.append(float(value_ms))

    def counter_total(self, name: str, **match: object) -> int:
        wanted = set(_labels(match))
        with self._lock:
            return sum(
                value
                for (counter, labels), value in self._counters.items()
                if counter == name and wanted <= set(labels)
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            counters = [
                {"name": name, "labels": dict(labels), "value": value}
                for (name, labels), value in sorted(self._counters.items())
            ]
            timings = [
                {"name": name, "labels": dict(labels), **summarize(samples)}
                for (name, labels), samples in sorted(self._samples.items())
            ]
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "counters": counters,
            "timings_ms": timings,
        }

    def prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            counter_names = sorted({name for name, _ in self._counters})
            for name in counter_names:
                metric = f"groundedai_{name}_total"
                lines.append(f"# TYPE {metric} counter")
                for (counter, labels), count in sorted(self._counters.items()):
                    if counter == name:
                        lines.append(f"{metric}{_format_labels(labels)} {count}")
            sample_names = sorted({name for name, _ in self._samples})
            for name in sample_names:
                metric = f"groundedai_{name}_ms"
                lines.append(f"# TYPE {metric} summary")
                for (sample_name, labels), samples in sorted(self._samples.items()):
                    if sample_name != name:
                        continue
                    data = list(samples)
                    for quantile in (0.5, 0.95):
                        value = percentile(data, quantile * 100) or 0.0
                        quantile_labels = (*labels, ("quantile", str(quantile)))
                        lines.append(f"{metric}{_format_labels(quantile_labels)} {value:.3f}")
                    lines.append(f"{metric}_count{_format_labels(labels)} {len(data)}")
                    lines.append(f"{metric}_sum{_format_labels(labels)} {sum(data):.3f}")
        lines.append("# TYPE groundedai_uptime_seconds gauge")
        lines.append(f"groundedai_uptime_seconds {time.time() - self.started_at:.1f}")
        return "\n".join(lines) + "\n"


def _format_labels(labels: LabelSet) -> str:
    if not labels:
        return ""
    escaped = ",".join(
        f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for key, value in labels
    )
    return "{" + escaped + "}"
