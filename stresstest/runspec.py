"""What a single run is: load shape, duration, and which part counts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Phase:
    """A stretch of a run with its own intensity.

    ``measure`` marks the part that ends up in the statistics. A warm-up
    phase is not measured -- the first minute of any run is dominated by
    cold prefills and would drag every percentile up -- except in the
    cold-start scenario, where the cold prefill *is* the measurement.
    """
    name: str
    duration_s: float
    intensity: float = 1.0
    active_fraction: float = 1.0   # 0.0 = everyone idle (classroom explanation)
    measure: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "duration_s": self.duration_s,
                "intensity": self.intensity, "active_fraction": self.active_fraction,
                "measure": self.measure}


@dataclass
class RunSpec:
    run_id: str
    label: str
    kind: str                      # sweep | activity | shared | engine | scenario | rampup | lesson
    students: int
    context_tokens: int
    activity: str = "normaal"
    shared_fraction: float = 0.5
    phases: list[Phase] = field(default_factory=list)
    arrival_window_s: float = 60.0   # how long the class takes to get going
    expects_shared_prefix: bool = True
    ramp: dict[str, Any] | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def total_duration_s(self) -> float:
        return sum(p.duration_s for p in self.phases)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "kind": self.kind,
            "students": self.students,
            "context_tokens": self.context_tokens,
            "activity": self.activity,
            "shared_fraction": self.shared_fraction,
            "arrival_window_s": self.arrival_window_s,
            "expects_shared_prefix": self.expects_shared_prefix,
            "duration_s": self.total_duration_s,
            "phases": [p.to_dict() for p in self.phases],
            "ramp": self.ramp,
            "tags": self.tags,
            "notes": self.notes,
        }


def standard_phases(warmup_s: float, measure_s: float, drain_s: float = 0.0) -> list[Phase]:
    phases = [Phase("warmup", warmup_s, measure=False),
              Phase("measure", measure_s, measure=True)]
    if drain_s > 0:
        phases.append(Phase("drain", drain_s, measure=False))
    return phases
