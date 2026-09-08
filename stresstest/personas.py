"""The behaviour model.

Twenty identical bots hammering an endpoint answer a question nobody asked.
Real students in a classroom behave very differently from each other, and
the peak load comes from several of them happening to be inside a burst at
the same moment -- not from a steady stream.

One student instruction ("build a login form") turns into 3-15 model calls
back to back: read file, edit, run tests, read output, fix. Between those
calls there is almost no pause. After the burst the student spends minutes
reading, thinking and typing. That shape is what this module produces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Persona:
    name: str
    share: float                       # fraction of the class
    think_time_s: tuple[float, float]  # roughly p10 and p90 of the pause between instructions
    steps_per_burst: tuple[int, int]   # model calls per instruction
    context_scale: tuple[float, float] # multiplier on the run's target context
    restart_probability: float = 0.0   # chance of throwing the session away and starting cold
    idle_probability: float = 0.0      # chance of skipping a turn entirely (the drop-out)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "share": self.share,
            "think_time_s": list(self.think_time_s),
            "steps_per_burst": list(self.steps_per_burst),
            "context_scale": list(self.context_scale),
            "restart_probability": self.restart_probability,
            "idle_probability": self.idle_probability,
        }


@dataclass
class StudentProfile:
    """One simulated student: a persona plus its own random stream."""
    index: int
    persona: Persona
    seed: int
    context_target_tokens: int
    rng: random.Random = field(repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random(self.seed)

    def next_think_time(self, intensity: float = 1.0) -> float:
        from .util import sample_lognormal
        low, high = self.persona.think_time_s
        value = sample_lognormal(self.rng, low, high)
        # Intensity > 1 means a busier moment (deadline, start of the lesson):
        # the same person simply pauses less.
        return max(0.5, value / max(intensity, 0.05))

    def next_burst_length(self, intensity: float = 1.0) -> int:
        low, high = self.persona.steps_per_burst
        base = self.rng.randint(low, high)
        scaled = int(round(base * min(max(intensity, 0.4), 2.0)))
        return max(1, scaled)

    def is_idle_this_turn(self) -> bool:
        return self.rng.random() < self.persona.idle_probability

    def wants_restart(self) -> bool:
        return self.rng.random() < self.persona.restart_probability


DEFAULT_PERSONAS: list[Persona] = [
    Persona("doorpakker", 0.15, (20, 60), (8, 15), (0.80, 1.00), 0.02, 0.00),
    Persona("gemiddelde", 0.45, (60, 180), (3, 8), (0.45, 0.70), 0.05, 0.02),
    Persona("worstelaar", 0.25, (120, 480), (2, 4), (0.20, 0.40), 0.25, 0.05),
    Persona("afhaker", 0.15, (300, 1200), (1, 3), (0.12, 0.30), 0.15, 0.30),
]

# The activity levels re-weight the same four personas rather than inventing
# new ones: a quiet lesson has more strugglers and drop-outs, an intense one
# more finishers. The individual behaviour does not change, the mix does.
ACTIVITY_LEVELS: dict[str, dict[str, float]] = {
    "rustig":    {"doorpakker": 0.05, "gemiddelde": 0.30, "worstelaar": 0.35, "afhaker": 0.30},
    "normaal":   {"doorpakker": 0.15, "gemiddelde": 0.45, "worstelaar": 0.25, "afhaker": 0.15},
    "intensief": {"doorpakker": 0.40, "gemiddelde": 0.45, "worstelaar": 0.12, "afhaker": 0.03},
}

# On top of the mix shift, an activity level also compresses or stretches
# think time globally. "Intensief" is a deadline, not a different class.
ACTIVITY_INTENSITY: dict[str, float] = {
    "rustig": 0.7,
    "normaal": 1.0,
    "intensief": 1.6,
}


def personas_from_config(config: Sequence[dict] | None) -> list[Persona]:
    if not config:
        return list(DEFAULT_PERSONAS)
    personas = []
    for entry in config:
        personas.append(Persona(
            name=entry["name"],
            share=float(entry["share"]),
            think_time_s=tuple(entry["think_time_s"]),          # type: ignore[arg-type]
            steps_per_burst=tuple(entry["steps_per_burst"]),    # type: ignore[arg-type]
            context_scale=tuple(entry.get("context_scale", [0.5, 0.8])),  # type: ignore[arg-type]
            restart_probability=float(entry.get("restart_probability", 0.0)),
            idle_probability=float(entry.get("idle_probability", 0.0)),
        ))
    return personas


def build_class(n_students: int,
                personas: Sequence[Persona],
                activity: str,
                context_tokens: int,
                seed: int,
                activity_mix: dict[str, dict[str, float]] | None = None) -> list[StudentProfile]:
    """Deterministically compose a class of ``n_students``.

    Largest-remainder allocation so that e.g. 20 students with the default
    mix give exactly 3 / 9 / 5 / 3 and not 3 / 9 / 5 / 2 plus a rounding
    hole. With the same seed you get the same class every time, which is
    what makes a run reproducible.
    """
    mix = (activity_mix or ACTIVITY_LEVELS).get(activity)
    weights = []
    for persona in personas:
        weights.append(mix.get(persona.name, persona.share) if mix else persona.share)
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]

    exact = [w * n_students for w in weights]
    counts = [int(value) for value in exact]
    remainder = n_students - sum(counts)
    order = sorted(range(len(personas)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1

    rng = random.Random(seed)
    profiles: list[StudentProfile] = []
    index = 0
    for persona, count in zip(personas, counts):
        for _ in range(count):
            low, high = persona.context_scale
            scale = rng.uniform(low, high)
            profiles.append(StudentProfile(
                index=index,
                persona=persona,
                seed=seed * 1000003 + index,
                context_target_tokens=max(1024, int(context_tokens * scale)),
            ))
            index += 1
    # Interleave so that student 0..n are not grouped by persona; matters for
    # scenarios that only start the first k students.
    rng.shuffle(profiles)
    for position, profile in enumerate(profiles):
        profile.index = position
    return profiles


def class_composition(profiles: Sequence[StudentProfile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for profile in profiles:
        counts[profile.persona.name] = counts.get(profile.persona.name, 0) + 1
    return counts
