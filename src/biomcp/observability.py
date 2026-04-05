from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Any

LabelTuple = tuple[tuple[str, str], ...]


def _normalize_labels(labels: Mapping[str, Any] | None = None) -> LabelTuple:
    if not labels:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


def _format_labels(labels: LabelTuple) -> str:
    if not labels:
        return ""
    pairs = []
    for key, value in labels:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        pairs.append(f'{key}="{escaped}"')
    return "{" + ",".join(pairs) + "}"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rank = max(0.0, min(1.0, q)) * (len(values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(values[low])
    weight = rank - low
    return float(values[low] * (1 - weight) + values[high] * weight)


class _MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, defaultdict[LabelTuple, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: dict[str, dict[LabelTuple, float]] = defaultdict(dict)
        self._histories: dict[str, defaultdict[LabelTuple, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=512))
        )
        self._help: dict[str, str] = {}
        self._types: dict[str, str] = {}

    def inc(
        self,
        name: str,
        value: float = 1.0,
        *,
        labels: Mapping[str, Any] | None = None,
        help_text: str = "",
    ) -> None:
        normalized = _normalize_labels(labels)
        with self._lock:
            self._types.setdefault(name, "counter")
            if help_text:
                self._help.setdefault(name, help_text)
            self._counters[name][normalized] += float(value)

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, Any] | None = None,
        help_text: str = "",
    ) -> None:
        normalized = _normalize_labels(labels)
        with self._lock:
            self._types.setdefault(name, "gauge")
            if help_text:
                self._help.setdefault(name, help_text)
            self._gauges[name][normalized] = float(value)

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, Any] | None = None,
        help_text: str = "",
    ) -> None:
        normalized = _normalize_labels(labels)
        with self._lock:
            self._types.setdefault(name, "summary")
            if help_text:
                self._help.setdefault(name, help_text)
            self._histories[name][normalized].append(float(value))

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            metric_names = sorted(
                set(self._types) | set(self._counters) | set(self._gauges) | set(self._histories)
            )
            for name in metric_names:
                lines.append(f"# HELP {name} {self._help.get(name, name)}")
                lines.append(f"# TYPE {name} {self._types.get(name, 'gauge')}")
                for labels, value in sorted(self._counters.get(name, {}).items()):
                    lines.append(f"{name}{_format_labels(labels)} {value}")
                for labels, value in sorted(self._gauges.get(name, {}).items()):
                    lines.append(f"{name}{_format_labels(labels)} {value}")
                histories: dict[LabelTuple, deque[float]] = dict(self._histories.get(name, {}))
                for labels, samples in sorted(histories.items()):
                    ordered = sorted(samples)
                    base = dict(labels)
                    for quantile in (0.5, 0.95):
                        qlabels = dict(base)
                        qlabels["quantile"] = f"{quantile:.2f}"
                        lines.append(
                            f"{name}{_format_labels(_normalize_labels(qlabels))} {_quantile(ordered, quantile):.6f}"
                        )
                    lines.append(f"{name}_count{_format_labels(labels)} {len(ordered)}")
                    lines.append(f"{name}_sum{_format_labels(labels)} {sum(ordered):.6f}")
        return "\n".join(lines) + "\n"


_REGISTRY = _MetricsRegistry()


def record_http_request(path: str, method: str, status_code: int, auth_mode: str) -> None:
    _REGISTRY.inc(
        "biomcp_http_requests_total",
        labels={
            "path": path,
            "method": method.upper(),
            "status": str(status_code),
            "auth_mode": auth_mode,
        },
        help_text="HTTP requests handled by the BioMCP hosted endpoint.",
    )


def record_tool_call(tool_name: str, status: str, latency_s: float) -> None:
    _REGISTRY.inc(
        "biomcp_tool_calls_total",
        labels={"tool": tool_name, "status": status},
        help_text="Tool invocations by tool name and terminal status.",
    )
    _REGISTRY.observe(
        "biomcp_tool_latency_seconds",
        latency_s,
        labels={"tool": tool_name},
        help_text="Observed tool latency with p50/p95 exported as summary quantiles.",
    )


def record_cache_event(namespace: str, event: str) -> None:
    _REGISTRY.inc(
        "biomcp_cache_events_total",
        labels={"namespace": namespace, "event": event},
        help_text="Cache hits, misses, and sets by cache namespace.",
    )


def record_upstream_request(host: str, status_code: int, latency_s: float) -> None:
    status_class = f"{status_code // 100}xx"
    _REGISTRY.inc(
        "biomcp_upstream_http_requests_total",
        labels={"host": host, "status_class": status_class},
        help_text="Upstream HTTP requests grouped by host and response status class.",
    )
    _REGISTRY.observe(
        "biomcp_upstream_latency_seconds",
        latency_s,
        labels={"host": host},
        help_text="Observed upstream request latency with p50/p95 exported as summary quantiles.",
    )


def record_upstream_error(host: str, error_type: str) -> None:
    _REGISTRY.inc(
        "biomcp_upstream_errors_total",
        labels={"host": host or "unknown", "error_type": error_type},
        help_text="Upstream transport or HTTP errors by host and exception class.",
    )


def record_auth_event(event: str, auth_mode: str = "oauth") -> None:
    _REGISTRY.inc(
        "biomcp_auth_events_total",
        labels={"event": event, "auth_mode": auth_mode},
        help_text="Authentication lifecycle events.",
    )


def set_runtime_gauge(name: str, value: float, *, labels: Mapping[str, Any] | None = None, help_text: str = "") -> None:
    _REGISTRY.set_gauge(name, value, labels=labels, help_text=help_text)


def render_prometheus_metrics() -> str:
    set_runtime_gauge(
        "biomcp_process_time_seconds",
        time.time(),
        help_text="Current wall-clock time when metrics were rendered.",
    )
    return _REGISTRY.render_prometheus()


__all__ = [
    "record_auth_event",
    "record_cache_event",
    "record_http_request",
    "record_tool_call",
    "record_upstream_error",
    "record_upstream_request",
    "render_prometheus_metrics",
    "set_runtime_gauge",
]
