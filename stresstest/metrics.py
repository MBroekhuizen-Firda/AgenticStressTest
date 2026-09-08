"""Client-side measurement.

Two levels matter and only one of them shows up in normal benchmarks:

*  **per request** -- time to first token, decode speed, errors. This is what
   the model does.
*  **per instruction (burst)** -- the wall-clock time from the student
   pressing enter until the agent has finished all 3-15 of its model calls.
   This is what the student actually experiences, and it is the number that
   decides whether a lesson works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .util import percentile, summarize


@dataclass
class RequestRecord:
    run_id: str
    student_index: int
    persona: str
    burst_index: int
    step_index: int
    start_t: float           # monotonic, relative to run start
    end_t: float
    ttft_s: float
    total_s: float
    connect_s: float
    prompt_tokens: int
    output_tokens: int
    reported_prompt_tokens: int | None
    cached_prompt_tokens: int | None
    decode_tokens_per_s: float
    itl_p50_s: float
    ok: bool
    error: str | None
    phase: str               # "warmup" | "measure" | "drain"

    def to_row(self) -> dict:
        return {
            "run_id": self.run_id,
            "student": self.student_index,
            "persona": self.persona,
            "burst": self.burst_index,
            "step": self.step_index,
            "start_s": round(self.start_t, 3),
            "end_s": round(self.end_t, 3),
            "ttft_s": self.ttft_s,
            "total_s": self.total_s,
            "connect_s": self.connect_s,
            "prompt_tokens": self.prompt_tokens,
            "reported_prompt_tokens": self.reported_prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "output_tokens": self.output_tokens,
            "decode_tps": self.decode_tokens_per_s,
            "itl_p50_s": self.itl_p50_s,
            "ok": int(self.ok),
            "error": self.error or "",
            "phase": self.phase,
        }


@dataclass
class BurstRecord:
    """One student instruction, from enter to the agent's last token."""
    run_id: str
    student_index: int
    persona: str
    burst_index: int
    start_t: float
    end_t: float
    steps: int
    failed_steps: int
    phase: str
    prompt_tokens_at_end: int

    @property
    def duration_s(self) -> float:
        return self.end_t - self.start_t

    def to_row(self) -> dict:
        return {
            "run_id": self.run_id,
            "student": self.student_index,
            "persona": self.persona,
            "burst": self.burst_index,
            "start_s": round(self.start_t, 3),
            "duration_s": round(self.duration_s, 3),
            "steps": self.steps,
            "failed_steps": self.failed_steps,
            "context_tokens": self.prompt_tokens_at_end,
            "phase": self.phase,
        }


@dataclass
class Collector:
    run_id: str
    requests: list[RequestRecord] = field(default_factory=list)
    bursts: list[BurstRecord] = field(default_factory=list)

    def add_request(self, record: RequestRecord) -> None:
        self.requests.append(record)

    def add_burst(self, record: BurstRecord) -> None:
        self.bursts.append(record)

    # ------------------------------------------------------------ aggregate

    def measured_requests(self) -> list[RequestRecord]:
        return [r for r in self.requests if r.phase == "measure"]

    def measured_bursts(self) -> list[BurstRecord]:
        return [b for b in self.bursts if b.phase == "measure"]

    def aggregate(self, window_s: float) -> dict:
        requests = self.measured_requests()
        good = [r for r in requests if r.ok]
        bursts = self.measured_bursts()

        ttft = [r.ttft_s for r in good if r.ttft_s == r.ttft_s]
        decode = [r.decode_tokens_per_s for r in good
                  if r.decode_tokens_per_s == r.decode_tokens_per_s]
        burst_durations = [b.duration_s for b in bursts]

        out: dict = {
            "requests_total": len(requests),
            "requests_ok": len(good),
            "requests_failed": len(requests) - len(good),
            "error_rate": (len(requests) - len(good)) / len(requests) if requests else 0.0,
            "bursts_completed": len(bursts),
            "ttft": summarize(ttft),
            "burst_duration": summarize(burst_durations),
            "decode_tps": summarize(decode),
            "request_total_s": summarize([r.total_s for r in good]),
            "output_tokens_total": sum(r.output_tokens for r in good),
            "prompt_tokens_total": sum(r.prompt_tokens for r in good),
            "client_output_tokens_per_s": (sum(r.output_tokens for r in good) / window_s)
            if window_s > 0 else float("nan"),
            "requests_per_minute": (len(requests) / window_s * 60.0) if window_s > 0 else float("nan"),
        }

        cached = [(r.cached_prompt_tokens, r.reported_prompt_tokens) for r in good
                  if r.cached_prompt_tokens is not None and r.reported_prompt_tokens]
        if cached:
            total_cached = sum(c for c, _ in cached)
            total_prompt = sum(p for _, p in cached)
            if total_prompt:
                out["client_cached_prompt_fraction"] = total_cached / total_prompt

        by_persona: dict[str, dict] = {}
        for persona in sorted({r.persona for r in requests}):
            subset = [r for r in good if r.persona == persona]
            persona_bursts = [b.duration_s for b in bursts if b.persona == persona]
            by_persona[persona] = {
                "requests": len(subset),
                "ttft_p90": percentile([r.ttft_s for r in subset if r.ttft_s == r.ttft_s], 90),
                "burst_p90": percentile(persona_bursts, 90),
                "decode_tps_p50": percentile(
                    [r.decode_tokens_per_s for r in subset
                     if r.decode_tokens_per_s == r.decode_tokens_per_s], 50),
            }
        out["by_persona"] = by_persona

        errors: dict[str, int] = {}
        for record in requests:
            if not record.ok:
                key = (record.error or "unknown").split(":")[0][:60]
                errors[key] = errors.get(key, 0) + 1
        out["errors"] = errors
        return out


def rolling_ttft_p90(records: Sequence[RequestRecord], since_t: float) -> float:
    """Used by the ramp-up cliff finder to grade the last slice of a run."""
    values = [r.ttft_s for r in records
              if r.ok and r.start_t >= since_t and r.ttft_s == r.ttft_s]
    return percentile(values, 90)
