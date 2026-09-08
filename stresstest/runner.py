"""Executing one run.

A run is a fixed-length simulation of a class. Students are asyncio tasks;
each one alternates between thinking (a long, skewed pause) and bursting
(3-15 model calls back to back). The run has phases, and only the phase
marked ``measure`` ends up in the statistics.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .client import OpenAIClient
from .conversation import TOOLS, Session
from .corpus import CodeCorpus
from .grading import Grade, grade_run, grade_run_brief_definition
from .metrics import BurstRecord, Collector, RequestRecord
from .personas import StudentProfile, build_class, class_composition, personas_from_config
from .runspec import Phase, RunSpec
from .tokens import TokenCounter
from .util import human_duration, log, now, percentile, wall
from .vllm_metrics import MetricsSampler


@dataclass
class RunResult:
    spec: RunSpec
    started_wall: float
    finished_wall: float
    aggregate: dict
    server: dict
    grade: Grade
    grade_brief: Grade
    composition: dict[str, int]
    collector: Collector = field(repr=False, default=None)  # type: ignore[assignment]
    metric_rows: list[dict] = field(repr=False, default_factory=list)
    ramp: dict | None = None

    def summary_row(self) -> dict:
        aggregate, server = self.aggregate, self.server
        return {
            "run_id": self.spec.run_id,
            "label": self.spec.label,
            "kind": self.spec.kind,
            "students": self.spec.students,
            "context_tokens": self.spec.context_tokens,
            "activity": self.spec.activity,
            "shared_fraction": self.spec.shared_fraction,
            "grade": self.grade.colour,
            "grade_brief": self.grade_brief.colour,
            "ttft_p50_s": aggregate.get("ttft", {}).get("p50"),
            "ttft_p90_s": aggregate.get("ttft", {}).get("p90"),
            "ttft_p99_s": aggregate.get("ttft", {}).get("p99"),
            "burst_p50_s": aggregate.get("burst_duration", {}).get("p50"),
            "burst_p90_s": aggregate.get("burst_duration", {}).get("p90"),
            "decode_tps_p50": aggregate.get("decode_tps", {}).get("p50"),
            "requests": aggregate.get("requests_total"),
            "requests_failed": aggregate.get("requests_failed"),
            "error_rate": aggregate.get("error_rate"),
            "bursts": aggregate.get("bursts_completed"),
            "prefix_cache_hit_rate": server.get("prefix_cache_hit_rate"),
            "preemptions": server.get("preemptions"),
            "kv_cache_usage_avg": server.get("kv_cache_usage_avg"),
            "kv_cache_usage_peak": server.get("kv_cache_usage_peak"),
            "queue_depth_avg": server.get("queue_depth_avg"),
            "queue_depth_peak": server.get("queue_depth_peak"),
            "server_prefill_tps": server.get("server_prefill_tokens_per_s"),
            "server_decode_tps": server.get("server_decode_tokens_per_s"),
            "client_output_tps": aggregate.get("client_output_tokens_per_s"),
            "max_students_ok": (self.ramp or {}).get("max_students_ok"),
            "reasons": "; ".join(self.grade.reasons),
            "warnings": "; ".join(self.grade.warnings),
            "duration_s": round(self.finished_wall - self.started_wall, 1),
        }


class _Clock:
    """Run-relative time plus which phase we are in."""

    def __init__(self, phases: Sequence[Phase]) -> None:
        self.phases = list(phases)
        self.origin = now()
        self.total = sum(p.duration_s for p in self.phases)

    @property
    def elapsed(self) -> float:
        return now() - self.origin

    def phase_at(self, t: float) -> Phase:
        cursor = 0.0
        for phase in self.phases:
            cursor += phase.duration_s
            if t < cursor:
                return phase
        return self.phases[-1] if self.phases else Phase("measure", 0, measure=True)

    def current(self) -> Phase:
        return self.phase_at(self.elapsed)

    def measure_window(self) -> tuple[float, float]:
        """Absolute monotonic bounds of the measured part of the run."""
        cursor, start, end = 0.0, None, None
        for phase in self.phases:
            if phase.measure and start is None:
                start = cursor
            if phase.measure:
                end = cursor + phase.duration_s
            cursor += phase.duration_s
        if start is None:
            start, end = 0.0, self.total
        return self.origin + start, self.origin + (end or self.total)


class RunEngine:
    def __init__(self, *, client: OpenAIClient, corpus: CodeCorpus,
                 counter: TokenCounter, config: dict) -> None:
        self.client = client
        self.corpus = corpus
        self.counter = counter
        self.config = config
        self.request_config = config.get("request", {})

    # ------------------------------------------------------------------ run

    async def run(self, spec: RunSpec, sampler: MetricsSampler,
                  progress: bool = True) -> RunResult:
        seed = int(self.config.get("seed", 20250908))
        personas = personas_from_config(self.config.get("behaviour", {}).get("personas"))
        activity_mix = self.config.get("behaviour", {}).get("activity_levels")
        profiles = build_class(spec.students, personas, spec.activity,
                               spec.context_tokens, seed, activity_mix)

        collector = Collector(spec.run_id)
        clock = _Clock(spec.phases)
        stop = asyncio.Event()
        state = _RunState(spec=spec, clock=clock, collector=collector, stop=stop)

        started_wall = wall()
        await sampler.start()

        tasks = [asyncio.create_task(self._student(profile, state))
                 for profile in profiles]
        if spec.ramp:
            tasks.append(asyncio.create_task(self._ramp_controller(state, profiles)))
        if progress:
            tasks.append(asyncio.create_task(self._progress(state, sampler)))

        try:
            await asyncio.wait_for(stop.wait(), timeout=clock.total + 5)
        except asyncio.TimeoutError:
            pass
        stop.set()

        drain_s = float(self.config.get("run_defaults", {}).get("drain_s", 45))
        done, pending = await asyncio.wait(tasks, timeout=drain_s)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            exc = task.exception() if not task.cancelled() else None
            if exc is not None:
                log(f"taak mislukt: {type(exc).__name__}: {exc}", color="red")
        await sampler.stop()
        finished_wall = wall()

        m_start, m_end = clock.measure_window()
        window_s = max(m_end - m_start, 1e-6)
        aggregate = collector.aggregate(window_s)
        server = sampler.summarize(m_start, m_end)
        grade = grade_run(aggregate, server,
                          self.config.get("grading", {}).get("thresholds"),
                          expects_shared_prefix=spec.expects_shared_prefix)
        grade_brief = grade_run_brief_definition(aggregate, server)

        return RunResult(
            spec=spec, started_wall=started_wall, finished_wall=finished_wall,
            aggregate=aggregate, server=server, grade=grade, grade_brief=grade_brief,
            composition=class_composition(profiles), collector=collector,
            metric_rows=sampler.rows(clock.origin),
            ramp=state.ramp_result if spec.ramp else None,
        )

    # -------------------------------------------------------------- student

    async def _student(self, profile: StudentProfile, state: "_RunState") -> None:
        spec, clock, collector = state.spec, state.clock, state.collector
        rng = random.Random(profile.seed ^ 0x5EED)
        session = Session(
            student_index=profile.index,
            corpus=self.corpus,
            counter=self.counter,
            target_tokens=profile.context_target_tokens,
            shared_fraction=spec.shared_fraction,
            rng=rng,
            message_style="tool_calls" if self.client.supports_tool_messages else "text",
        )
        # The class does not arrive in one instant, except in the cold-start
        # scenario where a narrow arrival window is the whole point.
        if spec.ramp:
            await state.wait_for_activation(profile.index)
            if state.stop.is_set():
                return
        else:
            delay = rng.uniform(0, max(spec.arrival_window_s, 0.01))
            if not await _sleep_until(delay, state.stop):
                return

        # Built after the stagger, not before: with an exact tokenizer a 100k
        # session costs ~150 ms to assemble, and thirty of those back to back
        # would block the event loop for four seconds right at the start of
        # the run -- exactly where we are trying to measure a cold start.
        session.reset()

        while not state.stop.is_set():
            phase = clock.current()
            if phase.active_fraction <= 0.0:
                # Classroom explanation / the ten minutes of silence.
                if not await _sleep_until(2.0, state.stop):
                    return
                continue
            if phase.active_fraction < 1.0 and \
                    (profile.index % 100) >= int(phase.active_fraction * 100):
                if not await _sleep_until(3.0, state.stop):
                    return
                continue
            if profile.is_idle_this_turn():
                if not await _sleep_until(profile.next_think_time(phase.intensity), state.stop):
                    return
                continue

            await self._burst(profile, session, state, phase)
            if state.stop.is_set():
                return
            if profile.wants_restart():
                session.restarts += 1
                session.reset()
            think = profile.next_think_time(phase.intensity)
            if not await _sleep_until(think, state.stop):
                return

    async def _burst(self, profile: StudentProfile, session: Session,
                     state: "_RunState", phase: Phase) -> None:
        clock, collector, spec = state.clock, state.collector, state.spec
        session.start_burst(phase.intensity)
        steps = profile.next_burst_length(phase.intensity)
        burst_start = clock.elapsed
        burst_phase = phase.name
        failed = 0
        completed = 0

        for step in range(steps):
            if state.stop.is_set():
                break
            messages, max_tokens = session.next_step()
            prompt_tokens = session.prompt_tokens_estimate
            start_t = clock.elapsed
            state.in_flight += 1
            try:
                result = await self.client.chat_stream(
                    messages,
                    max_tokens=max_tokens,
                    temperature=float(self.request_config.get("temperature", 0.0)),
                    tools=TOOLS if session.message_style == "tool_calls" else None,
                    extra_body=self._extra_body(),
                )
            finally:
                state.in_flight -= 1
            end_t = clock.elapsed
            itl = percentile(result.inter_token_latencies, 50) if result.inter_token_latencies else float("nan")
            record = RequestRecord(
                run_id=spec.run_id, student_index=profile.index,
                persona=profile.persona.name, burst_index=session.burst_number,
                step_index=step, start_t=start_t, end_t=end_t,
                ttft_s=result.ttft_s, total_s=result.total_s, connect_s=result.connect_s,
                prompt_tokens=prompt_tokens, output_tokens=result.output_tokens,
                reported_prompt_tokens=result.reported_prompt_tokens,
                cached_prompt_tokens=result.cached_prompt_tokens,
                decode_tokens_per_s=result.decode_tokens_per_s, itl_p50_s=itl,
                ok=result.ok, error=result.error,
                phase=_phase_label(clock.phase_at(start_t)),
            )
            collector.add_request(record)
            state.requests_done += 1
            if not result.ok:
                failed += 1
                state.consecutive_errors += 1
                if state.consecutive_errors > 25:
                    log("25 opeenvolgende fouten -- run afgebroken", color="red")
                    state.stop.set()
                break
            state.consecutive_errors = 0
            completed += 1
            session.record_step(result.text)

        burst_end = clock.elapsed
        collector.add_burst(BurstRecord(
            run_id=spec.run_id, student_index=profile.index,
            persona=profile.persona.name, burst_index=session.burst_number,
            start_t=burst_start, end_t=burst_end,
            steps=completed, failed_steps=failed,
            # A burst spans phase boundaries; its midpoint decides which phase
            # it belongs to. Classifying on the start would throw away every
            # instruction that began during warm-up, which on a five-minute
            # measurement window is most of them.
            phase=_phase_label(clock.phase_at((burst_start + burst_end) / 2)),
            prompt_tokens_at_end=session.prompt_tokens_estimate,
        ))

    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if self.request_config.get("ignore_eos", True):
            # Makes the decode length of every step exactly max_tokens, so the
            # load a run puts on the GPU is reproducible instead of depending
            # on where the model happened to stop. Turn off for non-vLLM
            # servers; the client downgrades automatically if it is rejected.
            body["ignore_eos"] = True
        extra = self.request_config.get("extra_body")
        if isinstance(extra, dict):
            body.update(extra)
        return body

    # ----------------------------------------------------------- cliff mode

    async def _ramp_controller(self, state: "_RunState",
                               profiles: Sequence[StudentProfile]) -> None:
        """Add one student every ``step_interval_s`` until the thresholds break.

        One run that answers 'how many students fit' directly, and it is
        reusable against any hardware configuration without changing anything.
        """
        ramp = state.spec.ramp or {}
        start_students = int(ramp.get("start_students", 5))
        interval = float(ramp.get("step_interval_s", 120))
        max_students = int(ramp.get("max_students", len(profiles)))
        thresholds = dict(self.config.get("grading", {}).get("thresholds") or {})
        ttft_limit = float(ramp.get("ttft_p90_limit_s")
                           or thresholds.get("ttft_p90_amber_s") or 45.0)
        burst_limit = float(ramp.get("burst_p90_limit_s")
                            or thresholds.get("burst_p90_amber_s") or 180.0)
        error_limit = float(ramp.get("error_rate_limit") or 0.02)

        state.activate_through(start_students)
        log(f"klifzoeker: start met {start_students} studenten, +1 per {interval:.0f}s",
            color="bold")
        steps: list[dict] = []
        last_ok = start_students
        active = start_students

        while not state.stop.is_set() and active <= max_students:
            if not await _sleep_until(interval, state.stop):
                break
            # Look back over at least a minute: with realistic think times a
            # shorter window can contain too few completed instructions to
            # judge anything, and an empty window must never read as "fine".
            window = max(interval, 60.0)
            since = state.clock.elapsed - window
            records = [r for r in state.collector.requests
                       if r.start_t >= since and r.ok and r.ttft_s == r.ttft_s]
            bursts = [b for b in state.collector.bursts if b.start_t >= since]
            all_records = [r for r in state.collector.requests if r.start_t >= since]
            ttft_p90 = percentile([r.ttft_s for r in records], 90)
            burst_p90 = percentile([b.duration_s for b in bursts], 90)
            errors = (len([r for r in all_records if not r.ok]) / len(all_records)
                      if all_records else 0.0)
            broke = []
            if ttft_p90 == ttft_p90 and ttft_p90 > ttft_limit:
                broke.append(f"p90 TTFT {ttft_p90:.1f}s > {ttft_limit:.0f}s")
            if burst_p90 == burst_p90 and burst_p90 > burst_limit:
                broke.append(f"p90 instructie {burst_p90:.0f}s > {burst_limit:.0f}s")
            if errors > error_limit:
                broke.append(f"foutpercentage {errors:.1%}")
            # Thin in either dimension: too few first tokens to trust a p90,
            # or too few completed instructions to say anything about them.
            thin = len(records) < 5 or len(bursts) < 3
            steps.append({"students": active, "ttft_p90_s": ttft_p90,
                          "burst_p90_s": burst_p90, "error_rate": errors,
                          "samples": len(records), "bursts": len(bursts),
                          "thin_sample": thin, "broke": broke})
            if thin and not broke:
                log(f"klifzoeker: bij {active} studenten maar {len(records)} "
                    f"verzoeken en {len(bursts)} instructies in het venster -- "
                    f"vergroot step_interval_s of verlaag de denktijden",
                    color="amber")
            colour = "red" if broke else "green"
            log(f"klifzoeker: {active:2d} studenten | p90 TTFT "
                f"{_seconds(ttft_p90):>7} | p90 instructie {_seconds(burst_p90):>7} | "
                f"{'GEBROKEN: ' + ', '.join(broke) if broke else 'ok'}", color=colour)
            if broke:
                state.ramp_result = {"max_students_ok": last_ok, "broke_at": active,
                                     "reasons": broke, "steps": steps}
                state.stop.set()
                return
            last_ok = active
            if active >= max_students:
                break
            active += 1
            state.activate_through(active)

        state.ramp_result = {"max_students_ok": last_ok, "broke_at": None,
                             "reasons": [], "steps": steps}
        state.stop.set()

    # -------------------------------------------------------------- display

    async def _progress(self, state: "_RunState", sampler: MetricsSampler) -> None:
        spec, clock = state.spec, state.clock
        interval = float(self.config.get("run_defaults", {}).get("progress_interval_s", 15))
        while not state.stop.is_set():
            if not await _sleep_until(interval, state.stop):
                return
            elapsed = clock.elapsed
            phase = clock.current()
            recent = [r for r in state.collector.requests if r.start_t >= elapsed - 60 and r.ok]
            ttft_p90 = percentile([r.ttft_s for r in recent if r.ttft_s == r.ttft_s], 90)
            latest = sampler.samples[-1].values if sampler.samples else {}
            bits = [
                f"{spec.run_id}",
                f"{human_duration(elapsed)}/{human_duration(clock.total)}",
                f"fase={phase.name}",
                f"verzoeken={state.requests_done}",
                f"actief={state.in_flight}",
                f"p90 TTFT(60s)={ttft_p90:5.1f}s" if ttft_p90 == ttft_p90 else "p90 TTFT(60s)=  n/b",
            ]
            if "kv_cache_usage" in latest:
                bits.append(f"KV={latest['kv_cache_usage']:.0%}")
            if "requests_waiting" in latest:
                bits.append(f"wachtrij={latest['requests_waiting']:.0f}")
            if "preemptions" in latest:
                bits.append(f"preempties={latest['preemptions']:.0f}")
            log("  " + " | ".join(bits), color="grey")


@dataclass
class _RunState:
    spec: RunSpec
    clock: _Clock
    collector: Collector
    stop: asyncio.Event
    in_flight: int = 0
    requests_done: int = 0
    consecutive_errors: int = 0
    # Ramp runs admit students one at a time, so nobody is active until the
    # controller says so. Ordinary runs never consult this field.
    active_students: int = 0
    ramp_result: dict | None = None
    _activation: asyncio.Event = field(default_factory=asyncio.Event)

    def activate_through(self, count: int) -> None:
        self.active_students = count
        self._activation.set()
        self._activation = asyncio.Event()

    async def wait_for_activation(self, index: int) -> None:
        """Block until the ramp controller has admitted this student."""
        while not self.stop.is_set() and index >= self.active_students:
            waiter = asyncio.ensure_future(self._activation.wait())
            stopper = asyncio.ensure_future(self.stop.wait())
            try:
                await asyncio.wait([waiter, stopper], return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (waiter, stopper):
                    if not task.done():
                        task.cancel()


def _seconds(value: float) -> str:
    return "n/b" if value != value else f"{value:.1f}s"


def _phase_label(phase: Phase) -> str:
    return "measure" if phase.measure else phase.name


async def _sleep_until(seconds: float, stop: asyncio.Event) -> bool:
    """Sleep, but wake immediately when the run ends. False = run is over."""
    if seconds <= 0:
        return not stop.is_set()
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
        return False
    except asyncio.TimeoutError:
        return True
