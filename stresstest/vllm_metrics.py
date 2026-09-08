"""Prometheus scraping for vLLM.

vLLM exposes almost everything we need on ``/metrics``; the harness only has
to poll it and take deltas over the measurement window. Metric names moved
between the V0 and V1 engines, so every quantity is looked up through a list
of aliases and the sampler records which name it actually found.

The four numbers that decide the answer to this whole investigation:

  prefix cache hit rate  how much of the class's context is shared and warm
  preemptions            a session evicted from the KV cache; the next agent
                         step then has to re-prefill its entire context
  KV cache usage         how close to full the GPU memory pool runs
  queue depth            requests that are waiting rather than running
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from .client import http_get_text
from .util import log, now, wall

# name in this harness -> candidate metric names, most recent engine first
METRIC_ALIASES: dict[str, list[str]] = {
    "prefix_cache_queries": ["vllm:prefix_cache_queries_total",
                             "vllm:gpu_prefix_cache_queries_total"],
    "prefix_cache_hits": ["vllm:prefix_cache_hits_total",
                          "vllm:gpu_prefix_cache_hits_total"],
    "prefix_cache_hit_rate_gauge": ["vllm:gpu_prefix_cache_hit_rate"],
    "preemptions": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    "kv_cache_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "requests_running": ["vllm:num_requests_running"],
    "requests_waiting": ["vllm:num_requests_waiting"],
    "requests_swapped": ["vllm:num_requests_swapped"],
    "prompt_tokens": ["vllm:prompt_tokens_total"],
    "generation_tokens": ["vllm:generation_tokens_total"],
    "request_success": ["vllm:request_success_total"],
    "iteration_tokens": ["vllm:iteration_tokens_total_sum", "vllm:iteration_tokens_total"],
}

_SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)\s*$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum every sample of a metric across its label sets.

    Summing is right for this harness: vLLM labels by model name and we run
    exactly one model, so the sum is the value. Histogram ``_sum``/``_count``
    series are kept under their full names.
    """
    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if value != value:  # NaN
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def pick(totals: dict[str, float], key: str) -> tuple[str | None, float | None]:
    for candidate in METRIC_ALIASES.get(key, []):
        if candidate in totals:
            return candidate, totals[candidate]
    return None, None


@dataclass
class MetricSample:
    t_monotonic: float
    t_wall: float
    values: dict[str, float] = field(default_factory=dict)
    raw_names: dict[str, str] = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {"t_wall": self.t_wall, "t_monotonic": round(self.t_monotonic, 3)}
        row.update({k: v for k, v in self.values.items()})
        return row


class MetricsSampler:
    """Polls ``/metrics`` on a fixed interval for the length of a run."""

    def __init__(self, url: str | None, interval_s: float = 2.0,
                 verify_tls: bool = True) -> None:
        self.url = url
        self.interval_s = interval_s
        self.verify_tls = verify_tls
        self.samples: list[MetricSample] = []
        self.available = False
        self.error: str | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def check(self) -> bool:
        if not self.url:
            self.error = "no metrics_url configured"
            return False
        try:
            text = await http_get_text(self.url, timeout_s=10.0, verify_tls=self.verify_tls)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        totals = parse_prometheus(text)
        self.available = any(name.startswith("vllm:") for name in totals)
        if not self.available:
            self.error = "endpoint reachable but exposes no vllm: metrics"
        return self.available

    async def _sample_once(self) -> MetricSample | None:
        try:
            text = await http_get_text(self.url or "", timeout_s=10.0, verify_tls=self.verify_tls)
        except Exception:
            return None
        totals = parse_prometheus(text)
        sample = MetricSample(t_monotonic=now(), t_wall=wall())
        for key in METRIC_ALIASES:
            name, value = pick(totals, key)
            if value is not None:
                sample.values[key] = value
                sample.raw_names[key] = name or ""
        return sample

    async def _loop(self) -> None:
        while not self._stop.is_set():
            sample = await self._sample_once()
            if sample is not None:
                self.samples.append(sample)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass

    async def start(self) -> None:
        if not self.url:
            return
        if not self.available and not await self.check():
            log(f"metrics: {self.error} -- continuing without server metrics", color="amber")
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None

    # ------------------------------------------------------------------ read

    def window(self, start_t: float, end_t: float) -> list[MetricSample]:
        return [s for s in self.samples if start_t <= s.t_monotonic <= end_t]

    def delta(self, key: str, start_t: float, end_t: float) -> float | None:
        """Counter increase over a window."""
        samples = [s for s in self.window(start_t, end_t) if key in s.values]
        if len(samples) < 2:
            return None
        return float(samples[-1].values[key] - samples[0].values[key])

    def average(self, key: str, start_t: float, end_t: float) -> float | None:
        values = [s.values[key] for s in self.window(start_t, end_t) if key in s.values]
        if not values:
            return None
        return float(sum(values) / len(values))

    def peak(self, key: str, start_t: float, end_t: float) -> float | None:
        values = [s.values[key] for s in self.window(start_t, end_t) if key in s.values]
        return float(max(values)) if values else None

    def summarize(self, start_t: float, end_t: float) -> dict[str, Any]:
        """Everything the grader and the report need from the server side."""
        out: dict[str, Any] = {"metrics_available": self.available}
        if not self.available:
            out["metrics_error"] = self.error
            return out

        queries = self.delta("prefix_cache_queries", start_t, end_t)
        hits = self.delta("prefix_cache_hits", start_t, end_t)
        if queries and queries > 0 and hits is not None:
            out["prefix_cache_hit_rate"] = hits / queries
            out["prefix_cache_queries"] = queries
            out["prefix_cache_hits"] = hits
        else:
            gauge = self.average("prefix_cache_hit_rate_gauge", start_t, end_t)
            if gauge is not None:
                out["prefix_cache_hit_rate"] = gauge
                out["prefix_cache_hit_rate_source"] = "gauge"

        preemptions = self.delta("preemptions", start_t, end_t)
        if preemptions is not None:
            out["preemptions"] = preemptions

        for key, label in (("kv_cache_usage", "kv_cache_usage"),
                           ("requests_running", "requests_running"),
                           ("requests_waiting", "queue_depth")):
            avg = self.average(key, start_t, end_t)
            top = self.peak(key, start_t, end_t)
            if avg is not None:
                out[f"{label}_avg"] = avg
                out[f"{label}_peak"] = top

        duration = max(end_t - start_t, 1e-6)
        prompt = self.delta("prompt_tokens", start_t, end_t)
        generation = self.delta("generation_tokens", start_t, end_t)
        if prompt is not None:
            out["server_prefill_tokens_per_s"] = prompt / duration
            out["server_prompt_tokens"] = prompt
        if generation is not None:
            out["server_decode_tokens_per_s"] = generation / duration
            out["server_generation_tokens"] = generation
        return out

    def rows(self, start_t: float | None = None) -> list[dict]:
        """Time series for the KV-usage / cache-hit-rate charts."""
        origin = start_t if start_t is not None else (
            self.samples[0].t_monotonic if self.samples else 0.0)
        rows = []
        previous: MetricSample | None = None
        for sample in self.samples:
            row = {"t_s": round(sample.t_monotonic - origin, 2), "t_wall": sample.t_wall}
            for key, value in sample.values.items():
                row[key] = value
            if previous is not None:
                dq = sample.values.get("prefix_cache_queries", 0) - previous.values.get("prefix_cache_queries", 0)
                dh = sample.values.get("prefix_cache_hits", 0) - previous.values.get("prefix_cache_hits", 0)
                if dq > 0:
                    row["prefix_cache_hit_rate_window"] = dh / dq
            previous = sample
            rows.append(row)
        return rows
