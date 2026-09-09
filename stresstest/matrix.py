"""Building the list of runs from the configuration.

Deliberately not the full cross product. Every axis crossed with every other
axis is several hundred runs and a rented GPU bills by the hour, so the
sweep moves one variable at a time and the variations hang off the points
where something is expected to happen.
"""

from __future__ import annotations

from typing import Any

from .runspec import Phase, RunSpec, standard_phases


def _defaults(config: dict) -> dict:
    return config.get("run_defaults", {})


def _phases(config: dict, warmup_s: float | None = None,
            measure_s: float | None = None) -> list[Phase]:
    defaults = _defaults(config)
    return standard_phases(
        float(warmup_s if warmup_s is not None else defaults.get("warmup_s", 90)),
        float(measure_s if measure_s is not None else defaults.get("measure_s", 300)),
    )


def build_sweep(config: dict) -> list[RunSpec]:
    matrix = config.get("matrix", {}).get("sweep", {})
    if not matrix.get("enabled", True):
        return []
    students = matrix.get("students", [5, 10, 20, 30])
    contexts = matrix.get("context_tokens", [8000, 32000, 64000, 100000])
    activity = matrix.get("activity", "normaal")
    shared = float(matrix.get("shared_fraction", 0.5))
    specs = []
    for n in students:
        for context in contexts:
            specs.append(RunSpec(
                run_id=f"sweep_s{n}_c{context // 1000}k",
                label=f"{n} studenten, {context // 1000}k context",
                kind="sweep", students=int(n), context_tokens=int(context),
                activity=activity, shared_fraction=shared,
                phases=_phases(config),
                arrival_window_s=float(_defaults(config).get("arrival_window_s", 60)),
                tags={"axis": "sweep"},
            ))
    return specs


def build_activity_variation(config: dict) -> list[RunSpec]:
    block = config.get("matrix", {}).get("activity_variation", {})
    if not block.get("enabled", True):
        return []
    students = block.get("students", [10, 20])
    levels = block.get("levels", ["rustig", "intensief"])
    context = int(block.get("context_tokens", 32000))
    shared = float(block.get("shared_fraction",
                             config.get("matrix", {}).get("sweep", {}).get("shared_fraction", 0.5)))
    specs = []
    for n in students:
        for level in levels:
            specs.append(RunSpec(
                run_id=f"act_{level}_s{n}",
                label=f"{n} studenten, activiteit {level}",
                kind="activity", students=int(n), context_tokens=context,
                activity=level, shared_fraction=shared, phases=_phases(config),
                tags={"axis": "activity"},
            ))
    return specs


def build_shared_variation(config: dict) -> list[RunSpec]:
    block = config.get("matrix", {}).get("shared_variation", {})
    if not block.get("enabled", True):
        return []
    students = int(block.get("students", 20))
    contexts = block.get("context_tokens", [32000, 64000])
    fractions = block.get("shared_fractions", [0.0, 0.5, 0.9])
    specs = []
    for context in contexts:
        for fraction in fractions:
            specs.append(RunSpec(
                run_id=f"shared_{int(fraction * 100):02d}_c{int(context) // 1000}k",
                label=f"{students} studenten, {int(context) // 1000}k, "
                      f"{int(fraction * 100)}% gedeelde projectbasis",
                kind="shared", students=students, context_tokens=int(context),
                activity="normaal", shared_fraction=float(fraction),
                phases=_phases(config),
                expects_shared_prefix=fraction >= 0.4,
                tags={"axis": "shared_fraction"},
            ))
    return specs


def build_engine_variation(config: dict) -> list[RunSpec]:
    """vLLM flags cannot be changed from the client: the server has to be
    restarted. These runs are therefore emitted with the server command line
    they need, and the runner pauses for it (or you drive them one at a time
    with ``--only``)."""
    block = config.get("matrix", {}).get("engine_variation", {})
    if not block.get("enabled", True):
        return []
    students = int(block.get("students", 20))
    context = int(block.get("context_tokens", 32000))
    shared = float(block.get("shared_fraction", 0.5))
    specs = []
    for variant in block.get("variants", []):
        specs.append(RunSpec(
            run_id=f"engine_{variant['name']}",
            label=f"vLLM-variant {variant['name']}",
            kind="engine", students=students, context_tokens=context,
            activity="normaal", shared_fraction=shared, phases=_phases(config),
            tags={"axis": "engine", "server_flags": variant.get("flags", "")},
            notes=variant.get("note", ""),
        ))
    return specs


def build_scenarios(config: dict) -> list[RunSpec]:
    """The named situations. These matter more than the averages, because
    they are what goes wrong."""
    block = config.get("matrix", {}).get("scenarios", {})
    if not block.get("enabled", True):
        return []
    defaults = _defaults(config)
    warmup = float(defaults.get("warmup_s", 90))
    measure = float(defaults.get("measure_s", 300))
    class_size = int(block.get("class_size", 20))
    big_class = int(block.get("oversized_class", 30))
    context = int(block.get("context_tokens", 32000))
    specs: list[RunSpec] = []

    # Cold start: the teacher says "open your project". No cache hits at all,
    # everyone at once. Measured from the first second, because the cold
    # prefill IS the measurement here -- there is no warm-up to discard.
    for n in block.get("cold_start_students", [class_size, big_class]):
        specs.append(RunSpec(
            run_id=f"scen_koudestart_s{n}",
            label=f"Koude start, {n} studenten",
            kind="scenario", students=int(n),
            context_tokens=int(block.get("cold_start_context_tokens", 12000)),
            activity="normaal", shared_fraction=0.0,
            phases=[Phase("koude_start", float(block.get("cold_start_measure_s", 240)),
                          intensity=1.4, measure=True)],
            arrival_window_s=float(block.get("cold_start_arrival_s", 120)),
            expects_shared_prefix=False,
            tags={"scenario": "koude_start"},
            notes="Alle studenten vuren binnen twee minuten hun eerste, volledig koude "
                  "prefill af. Nul cachehits, allemaal tegelijk.",
        ))

    specs.append(RunSpec(
        run_id="scen_deadline",
        label=f"Deadline, {class_size} studenten in burst",
        kind="scenario", students=class_size, context_tokens=context,
        activity="intensief", shared_fraction=0.5,
        phases=[Phase("warmup", warmup, intensity=1.6),
                Phase("deadline", measure, intensity=2.0, measure=True)],
        arrival_window_s=20.0,
        tags={"scenario": "deadline"},
        notes="Laatste twintig minuten voor inleveren: iedereen tegelijk in burstmodus.",
    ))

    specs.append(RunSpec(
        run_id="scen_lange_sessies",
        label="Lange sessies, weinig studenten, zeer grote context",
        kind="scenario", students=int(block.get("long_session_students", 6)),
        context_tokens=int(block.get("long_session_context_tokens", 110000)),
        activity="intensief", shared_fraction=0.3,
        phases=_phases(config, warmup_s=warmup, measure_s=measure),
        tags={"scenario": "lange_sessies"},
        notes="Test geheugen in plaats van rekenkracht: de geheugenklif zonder "
              "ruis van gelijktijdigheid.",
    ))

    silence_s = float(block.get("silence_s", 600))
    specs.append(RunSpec(
        run_id="scen_na_de_stilte",
        label="Na de stilte: tien minuten niets, dan hervat iedereen tegelijk",
        kind="scenario", students=class_size, context_tokens=context,
        activity="normaal", shared_fraction=0.5,
        phases=[Phase("opbouw", float(block.get("buildup_s", 240))),
                Phase("stilte", silence_s, active_fraction=0.0),
                Phase("hervatten", float(block.get("resume_measure_s", 240)),
                      intensity=1.5, measure=True)],
        arrival_window_s=30.0,
        tags={"scenario": "na_de_stilte"},
        notes="Zijn de sessies uit de cache gegooid tijdens de klassikale uitleg, "
              "en hoe duur is de eerste stap daarna?",
    ))

    specs.append(RunSpec(
        run_id="scen_worst_case",
        label=f"Worst case: {class_size} studenten, maximale context, geen gedeelde prefix",
        kind="scenario", students=class_size,
        context_tokens=int(block.get("worst_case_context_tokens", 100000)),
        activity="intensief", shared_fraction=0.0,
        phases=[Phase("warmup", warmup, intensity=2.0),
                Phase("worst_case", measure, intensity=2.0, measure=True)],
        arrival_window_s=15.0, expects_shared_prefix=False,
        tags={"scenario": "worst_case"},
        notes="Niet realistisch, maar het legt de harde bovengrens vast.",
    ))
    return specs


def build_rampup(config: dict) -> list[RunSpec]:
    block = config.get("matrix", {}).get("rampup", {})
    if not block.get("enabled", True):
        return []
    start = int(block.get("start_students", 5))
    max_students = int(block.get("max_students", 40))
    interval = float(block.get("step_interval_s", 120))
    step = max(int(block.get("step_students", 1)), 1)
    # math.ceil without the import: how many steps to walk from start to max.
    steps = -(-(max_students - start) // step)
    duration = interval * (steps + 2)
    return [RunSpec(
        run_id="rampup",
        label=f"Klifzoeker: {start} studenten, +{step} per {int(interval)}s tot het breekt",
        kind="rampup", students=max_students,
        context_tokens=int(block.get("context_tokens", 32000)),
        activity=block.get("activity", "normaal"),
        shared_fraction=float(block.get("shared_fraction", 0.5)),
        phases=[Phase("ramp", duration, measure=True)],
        arrival_window_s=0.0,
        # Only put keys in here that carry a value: the controller falls back
        # to the grading thresholds for anything absent, and a None would
        # override that fallback with nothing.
        ramp={key: value for key, value in {
            "start_students": start,
            "max_students": max_students,
            "step_interval_s": interval,
            "step_students": step,
            "ttft_p90_limit_s": block.get("ttft_p90_limit_s"),
            "burst_p90_limit_s": block.get("burst_p90_limit_s"),
            "error_rate_limit": block.get("error_rate_limit", 0.02),
        }.items() if value is not None},
        tags={"axis": "rampup"},
        notes="Waarschijnlijk het meest bruikbare enkele getal dat de test oplevert.",
    )]


def build_lesson(config: dict) -> list[RunSpec]:
    """Phase 2: one realistic ninety-minute lesson."""
    block = config.get("lesson", {})
    students = int(block.get("students", 20))
    context = int(block.get("context_tokens", 32000))
    shared = float(block.get("shared_fraction", 0.5))
    scale = float(block.get("time_scale", 1.0))   # 0.5 halves the lesson, for a dry run
    phases = [
        Phase("opstartpiek", 8 * 60 * scale, intensity=1.5, measure=True),
        Phase("opbouw", 27 * 60 * scale, intensity=1.0, measure=True),
        Phase("klassikale_uitleg", 10 * 60 * scale, active_fraction=0.0),
        Phase("eindpiek", 35 * 60 * scale, intensity=1.6, measure=True),
        Phase("afbouw", 10 * 60 * scale, intensity=0.6, measure=True),
    ]
    return [RunSpec(
        run_id="les_90min",
        label=f"Lesvalidatie 90 minuten, {students} studenten",
        kind="lesson", students=students, context_tokens=context,
        activity="normaal", shared_fraction=shared, phases=phases,
        arrival_window_s=float(block.get("arrival_window_s", 120)),
        tags={"axis": "lesson"},
        notes="Fase 2: de koude start van een echte les plus wat er met de cache "
              "gebeurt tijdens tien minuten klassikale uitleg.",
    )]


BUILDERS = {
    "sweep": build_sweep,
    "activity": build_activity_variation,
    "shared": build_shared_variation,
    "engine": build_engine_variation,
    "scenarios": build_scenarios,
    "rampup": build_rampup,
    "lesson": build_lesson,
}

PHASE1 = ["sweep", "activity", "shared", "scenarios", "engine", "rampup"]


def build_specs(config: dict, groups: list[str] | None = None) -> list[RunSpec]:
    wanted = groups or PHASE1
    specs: list[RunSpec] = []
    for group in wanted:
        builder = BUILDERS.get(group)
        if builder is None:
            raise SystemExit(f"unknown run group {group!r}; known: {', '.join(BUILDERS)}")
        specs.extend(builder(config))
    return specs


def estimate_duration_s(specs: list[RunSpec], gap_s: float = 30.0) -> float:
    return sum(spec.total_duration_s + gap_s for spec in specs)


def describe_plan(specs: list[RunSpec], gap_s: float = 30.0) -> dict[str, Any]:
    by_kind: dict[str, dict] = {}
    for spec in specs:
        entry = by_kind.setdefault(spec.kind, {"runs": 0, "seconds": 0.0})
        entry["runs"] += 1
        entry["seconds"] += spec.total_duration_s + gap_s
    return {"runs": len(specs),
            "total_seconds": estimate_duration_s(specs, gap_s),
            "by_kind": by_kind}
