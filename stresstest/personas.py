"""The behaviour model.

Twenty identical bots hammering an endpoint answer a question nobody asked.
Real students in a classroom behave very differently from each other, and
the peak load comes from several of them happening to be inside a burst at
the same moment -- not from a steady stream.

One student instruction ("build a login form") turns into 3-15 model calls
back to back: read file, edit, run tests, read output, fix. Between those
calls there is almost no pause. After the burst the student spends minutes
reading, thinking and typing. That shape is what this module produces.

There are two independent axes, and conflating them was the original mistake.

*  A **persona** is a pace: how long someone thinks, how many steps one
   instruction takes, how much context they carry, whether they drop out.
*  A **work profile** is a weight: how big the codebase is, how many files
   the agent reads at once, how much output each step produces. A student
   tweaking a form and a student having the agent grind through a Unity
   project can both be "gemiddelde" personas.

They are drawn separately, so a class is a mix on both axes. The work profile
is what decides the load per model call, and it was previously a hard-coded
assumption -- one small file per read, a couple of hundred output tokens --
that turned out to be the least defensible number in the harness.
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
class WorkProfile:
    """How heavy one agent step is for this student.

    Every field here is something you could count in a recording of a real
    session, which is the point: these are meant to be calibrated against
    observation rather than guessed at once and then forgotten.
    """
    name: str
    share: float
    group: str | None = None                  # which corpus group / assignment
    files_per_read: tuple[int, int] = (1, 1)  # files returned by one read_file
    read_line_budget: tuple[int, int] = (0, 0)  # 0 = whole file, else line window
    search_matches: tuple[int, int] = (8, 8)
    search_context_lines: int = 0
    test_output_blocks: tuple[int, int] = (1, 1)  # failure blocks in one test run
    output_tokens_tool: tuple[int, int] = (60, 260)   # a step that calls a tool
    output_tokens_final: tuple[int, int] = (220, 700)  # the closing explanation
    tool_mix: dict[str, float] = field(default_factory=lambda: {
        "read_file": 0.45, "edit_file": 0.25, "run_tests": 0.20, "search_code": 0.10})

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "share": self.share,
            "group": self.group,
            "files_per_read": list(self.files_per_read),
            "read_line_budget": list(self.read_line_budget),
            "search_matches": list(self.search_matches),
            "search_context_lines": self.search_context_lines,
            "test_output_blocks": list(self.test_output_blocks),
            "output_tokens_tool": list(self.output_tokens_tool),
            "output_tokens_final": list(self.output_tokens_final),
            "tool_mix": dict(self.tool_mix),
        }


@dataclass
class StudentProfile:
    """One simulated student: a persona plus its own random stream."""
    index: int
    persona: Persona
    seed: int
    context_target_tokens: int
    work: "WorkProfile" = None  # type: ignore[assignment]
    rng: random.Random = field(repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random(self.seed)
        if self.work is None:
            self.work = DEFAULT_WORK_PROFILES[0]

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

# Three ways of working, drawn independently of the persona. The numbers are
# starting points, not measurements: "klein" is the behaviour the harness
# always had, "doorspitten" is what a session on a real codebase looks like,
# and "middel" sits between them. Recording real sessions and re-deriving
# these is the single most valuable calibration left to do -- until then the
# point of having three is that the matrix shows how much the answer moves.
DEFAULT_WORK_PROFILES: list["WorkProfile"] = [
    WorkProfile(
        name="klein", share=0.50, group=None,
        files_per_read=(1, 1), read_line_budget=(0, 0),
        search_matches=(4, 10), search_context_lines=0,
        test_output_blocks=(1, 1),
        output_tokens_tool=(60, 260), output_tokens_final=(220, 700),
        tool_mix={"read_file": 0.45, "edit_file": 0.25,
                  "run_tests": 0.20, "search_code": 0.10},
    ),
    WorkProfile(
        name="middel", share=0.30, group=None,
        files_per_read=(1, 3), read_line_budget=(0, 0),
        search_matches=(8, 20), search_context_lines=2,
        test_output_blocks=(1, 3),
        output_tokens_tool=(120, 500), output_tokens_final=(400, 1200),
        tool_mix={"read_file": 0.45, "edit_file": 0.25,
                  "run_tests": 0.20, "search_code": 0.10},
    ),
    WorkProfile(
        name="doorspitten", share=0.20, group=None,
        files_per_read=(2, 5), read_line_budget=(0, 0),
        search_matches=(15, 40), search_context_lines=3,
        test_output_blocks=(2, 5),
        output_tokens_tool=(250, 900), output_tokens_final=(800, 2500),
        # Grinding through an unfamiliar codebase is mostly reading and
        # searching, not editing.
        tool_mix={"read_file": 0.50, "edit_file": 0.15,
                  "run_tests": 0.15, "search_code": 0.20},
    ),
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


def work_profiles_from_config(config: Sequence[dict] | None) -> list[WorkProfile]:
    if not config:
        return list(DEFAULT_WORK_PROFILES)
    out = []
    for entry in config:
        base = WorkProfile(name=entry["name"], share=float(entry["share"]))
        out.append(WorkProfile(
            name=base.name,
            share=base.share,
            group=entry.get("group"),
            files_per_read=tuple(entry.get("files_per_read", base.files_per_read)),  # type: ignore[arg-type]
            read_line_budget=tuple(entry.get("read_line_budget", base.read_line_budget)),  # type: ignore[arg-type]
            search_matches=tuple(entry.get("search_matches", base.search_matches)),  # type: ignore[arg-type]
            search_context_lines=int(entry.get("search_context_lines",
                                               base.search_context_lines)),
            test_output_blocks=tuple(entry.get("test_output_blocks", base.test_output_blocks)),  # type: ignore[arg-type]
            output_tokens_tool=tuple(entry.get("output_tokens_tool", base.output_tokens_tool)),  # type: ignore[arg-type]
            output_tokens_final=tuple(entry.get("output_tokens_final", base.output_tokens_final)),  # type: ignore[arg-type]
            tool_mix=dict(entry.get("tool_mix") or base.tool_mix),
        ))
    return out


def _allocate(n: int, weights: Sequence[float]) -> list[int]:
    """Largest-remainder allocation of ``n`` over ``weights``.

    Used for both axes: 20 students with the default persona mix must give
    exactly 3/9/5/3, not 3/9/5/2 plus a rounding hole.
    """
    total = sum(weights) or 1.0
    normalised = [w / total for w in weights]
    exact = [w * n for w in normalised]
    counts = [int(value) for value in exact]
    remainder = n - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def build_class(n_students: int,
                personas: Sequence[Persona],
                activity: str,
                context_tokens: int,
                seed: int,
                activity_mix: dict[str, dict[str, float]] | None = None,
                work_profiles: Sequence[WorkProfile] | None = None) -> list[StudentProfile]:
    """Deterministically compose a class of ``n_students``.

    Largest-remainder allocation so that e.g. 20 students with the default
    mix give exactly 3 / 9 / 5 / 3 and not 3 / 9 / 5 / 2 plus a rounding
    hole. With the same seed you get the same class every time, which is
    what makes a run reproducible.

    Personas and work profiles are allocated separately and then paired, so
    the two axes stay independent: the heavy-codebase students are not all
    finishers, and the drop-outs are not all on the small assignment.
    """
    mix = (activity_mix or ACTIVITY_LEVELS).get(activity)
    weights = []
    for persona in personas:
        weights.append(mix.get(persona.name, persona.share) if mix else persona.share)
    counts = _allocate(n_students, weights)

    work = list(work_profiles or DEFAULT_WORK_PROFILES)
    work_counts = _allocate(n_students, [w.share for w in work])
    work_slots: list[WorkProfile] = []
    for profile, count in zip(work, work_counts):
        work_slots.extend([profile] * count)

    rng = random.Random(seed)
    # Shuffled independently of the persona order, so pairing the two lists is
    # a random pairing rather than a correlated one.
    rng.shuffle(work_slots)

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
                work=work_slots[index] if index < len(work_slots) else work[0],
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


def work_composition(profiles: Sequence[StudentProfile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for profile in profiles:
        name = profile.work.name if profile.work else "onbekend"
        counts[name] = counts.get(name, 0) + 1
    return counts
