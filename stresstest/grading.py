"""Turning a run into a colour.

The brief proposed: green under 20 s p90 TTFT, no preemptions, prefix cache
hit rate above 60 %. Two adjustments, both argued in RESULTATEN.md:

*  **Burst duration and decode speed are added as criteria.** Time to first
   token is only the first of 3-15 model calls in one instruction. A run can
   have a fine TTFT and still take three minutes to finish one instruction,
   which is the thing the student actually waits for. Likewise, 20 s to the
   first token followed by 4 tokens/s is a bad experience that a TTFT-only
   rule would call green.
*  **Prefix cache hit rate is reported but is not a green criterion.** It is
   a property of the scenario, not of the hardware: in the deliberate 0 %
   shared-base runs and in the cold-start scenario a 60 % hit rate is
   unreachable by construction, and failing those runs on it would hide the
   real signal. A low hit rate where we expected a high one is flagged as a
   separate warning instead.

Preemptions stay a hard failure. Every preemption means a session was thrown
out of the KV cache and the next agent step has to re-prefill its whole
context -- the failure mode this whole investigation is looking for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GREEN, AMBER, RED = "groen", "oranje", "rood"
ORDER = {GREEN: 0, AMBER: 1, RED: 2}

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "ttft_p90_green_s": 20.0,
    "ttft_p90_amber_s": 45.0,
    "burst_p90_green_s": 90.0,
    "burst_p90_amber_s": 180.0,
    "decode_tps_p50_green": 12.0,
    "decode_tps_p50_amber": 6.0,
    "error_rate_green": 0.0,
    "error_rate_amber": 0.02,
    "preemptions_green": 0,
    "preemptions_amber": 5,
    "kv_usage_warn": 0.90,
    "prefix_cache_expectation": 0.60,
}


@dataclass
class Grade:
    colour: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    criteria: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"colour": self.colour, "reasons": self.reasons,
                "warnings": self.warnings, "criteria": self.criteria}


def _fmt(value: float, suffix: str = "", digits: int = 1) -> str:
    if value != value:
        return "niet gemeten"
    return f"{value:.{digits}f}{suffix}"


def _band(value: float, green_limit: float, amber_limit: float,
          higher_is_better: bool = False) -> str:
    if value != value:            # NaN: nothing measured
        return AMBER
    if higher_is_better:
        if value >= green_limit:
            return GREEN
        if value >= amber_limit:
            return AMBER
        return RED
    if value <= green_limit:
        return GREEN
    if value <= amber_limit:
        return AMBER
    return RED


def grade_run(aggregate: dict, server: dict, thresholds: dict | None = None,
              expects_shared_prefix: bool = True) -> Grade:
    limits = dict(DEFAULT_THRESHOLDS)
    limits.update(thresholds or {})
    grade = Grade(colour=GREEN)

    def apply(name: str, colour: str, message: str) -> None:
        grade.criteria[name] = colour
        if ORDER[colour] > ORDER[grade.colour]:
            grade.colour = colour
        if colour != GREEN:
            grade.reasons.append(message)

    ttft_p90 = aggregate.get("ttft", {}).get("p90", float("nan"))
    colour = _band(ttft_p90, limits["ttft_p90_green_s"], limits["ttft_p90_amber_s"])
    apply("ttft_p90", colour, f"p90 TTFT {_fmt(ttft_p90, 's')}")

    burst_p90 = aggregate.get("burst_duration", {}).get("p90", float("nan"))
    colour = _band(burst_p90, limits["burst_p90_green_s"], limits["burst_p90_amber_s"])
    apply("burst_p90", colour, f"p90 doorlooptijd instructie {_fmt(burst_p90, 's', 0)}")
    if burst_p90 != burst_p90:
        grade.warnings.append("geen enkele instructie volledig afgerond binnen het "
                              "meetvenster -- verleng measure_s of verlaag de denktijden")

    decode_p50 = aggregate.get("decode_tps", {}).get("p50", float("nan"))
    colour = _band(decode_p50, limits["decode_tps_p50_green"],
                   limits["decode_tps_p50_amber"], higher_is_better=True)
    apply("decode_tps_p50", colour, f"decodesnelheid {_fmt(decode_p50, ' tok/s per stream')}")

    error_rate = aggregate.get("error_rate", 0.0)
    colour = _band(error_rate, limits["error_rate_green"], limits["error_rate_amber"])
    apply("error_rate", colour, f"{error_rate:.1%} van de verzoeken faalde")

    preemptions = server.get("preemptions")
    if preemptions is None:
        grade.criteria["preemptions"] = "onbekend"
        grade.warnings.append("preempties niet gemeten (geen /metrics)")
    else:
        colour = _band(float(preemptions), float(limits["preemptions_green"]),
                       float(limits["preemptions_amber"]))
        apply("preemptions", colour, f"{int(preemptions)} preempties")

    kv_peak = server.get("kv_cache_usage_peak")
    if kv_peak is not None and kv_peak >= limits["kv_usage_warn"]:
        grade.warnings.append(f"KV-cache piekte op {kv_peak:.0%} -- weinig marge")

    hit_rate = server.get("prefix_cache_hit_rate")
    if hit_rate is not None:
        grade.criteria["prefix_cache_hit_rate"] = f"{hit_rate:.0%}"
        if expects_shared_prefix and hit_rate < limits["prefix_cache_expectation"]:
            grade.warnings.append(
                f"prefix cache hit rate {hit_rate:.0%} onder de verwachte "
                f"{limits['prefix_cache_expectation']:.0%}")

    queue_peak = server.get("queue_depth_peak")
    if queue_peak is not None and queue_peak > 0:
        grade.warnings.append(f"wachtrij piekte op {queue_peak:.0f} verzoeken")

    return grade


def grade_run_brief_definition(aggregate: dict, server: dict) -> Grade:
    """The original definition from the brief, kept so both can be compared."""
    grade = Grade(colour=GREEN)
    ttft_p90 = aggregate.get("ttft", {}).get("p90", float("nan"))
    if ttft_p90 != ttft_p90 or ttft_p90 > 45:
        grade.colour = RED
        grade.reasons.append(f"p90 TTFT {ttft_p90:.1f}s")
    elif ttft_p90 > 20:
        grade.colour = AMBER
        grade.reasons.append(f"p90 TTFT {ttft_p90:.1f}s")
    preemptions = server.get("preemptions")
    if preemptions:
        if preemptions > 5 and grade.colour != RED:
            grade.colour = RED
        elif ORDER[grade.colour] < ORDER[AMBER]:
            grade.colour = AMBER
        grade.reasons.append(f"{int(preemptions)} preempties")
    hit_rate = server.get("prefix_cache_hit_rate")
    if hit_rate is not None and hit_rate < 0.60 and grade.colour == GREEN:
        grade.colour = AMBER
        grade.reasons.append(f"cache hit rate {hit_rate:.0%}")
    if aggregate.get("error_rate", 0.0) > 0:
        grade.colour = RED
        grade.reasons.append("verzoeken liepen af")
    return grade
