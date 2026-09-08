"""Tests for the parts where a silent mistake would invalidate a whole run.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stresstest.conversation import Session
from stresstest.corpus import CodeCorpus, SourceFile
from stresstest.grading import grade_run
from stresstest.matrix import build_specs, describe_plan
from stresstest.personas import DEFAULT_PERSONAS, build_class, class_composition
from stresstest.tokens import build_counter
from stresstest.util import load_jsonc, percentile, sample_lognormal
from stresstest.vllm_metrics import parse_prometheus, pick

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_corpus(files: int = 60, shared: int = 20) -> CodeCorpus:
    sources = [
        SourceFile(f"app/module_{i:03d}.py",
                   "".join(f"def function_{i}_{j}(value):\n    return value + {j}\n\n"
                           for j in range(30)))
        for i in range(files)
    ]
    return CodeCorpus("test", sources, shared)


class TestStatistics(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(percentile([10], 90), 10.0)
        self.assertNotEqual(percentile([], 90), percentile([], 90))  # NaN

    def test_lognormal_is_skewed_and_bounded(self):
        rng = random.Random(7)
        draws = [sample_lognormal(rng, 60, 180) for _ in range(4000)]
        median = percentile(draws, 50)
        self.assertLess(abs(median - 104), 15, "median should sit near sqrt(60*180)")
        self.assertGreater(percentile(draws, 99), percentile(draws, 90),
                           "distribution must have a tail")
        self.assertTrue(all(d > 0 for d in draws))


class TestClassComposition(unittest.TestCase):
    def test_counts_add_up_exactly(self):
        for size in (1, 5, 7, 10, 20, 30, 41):
            profiles = build_class(size, DEFAULT_PERSONAS, "normaal", 32000, 42)
            self.assertEqual(len(profiles), size)
            self.assertEqual(sum(class_composition(profiles).values()), size)

    def test_same_seed_gives_same_class(self):
        a = build_class(20, DEFAULT_PERSONAS, "normaal", 32000, 42)
        b = build_class(20, DEFAULT_PERSONAS, "normaal", 32000, 42)
        self.assertEqual([p.persona.name for p in a], [p.persona.name for p in b])
        self.assertEqual([p.context_target_tokens for p in a],
                         [p.context_target_tokens for p in b])

    def test_activity_shifts_the_mix(self):
        quiet = class_composition(build_class(20, DEFAULT_PERSONAS, "rustig", 32000, 1))
        busy = class_composition(build_class(20, DEFAULT_PERSONAS, "intensief", 32000, 1))
        self.assertGreater(busy.get("doorpakker", 0), quiet.get("doorpakker", 0))
        self.assertGreater(quiet.get("afhaker", 0), busy.get("afhaker", 0))


class TestConversation(unittest.TestCase):
    """The shared prefix is the single most important property of the harness:
    if students do not share a byte-identical prefix, the whole prefix-cache
    axis of the experiment measures nothing."""

    def setUp(self):
        self.corpus = make_corpus()
        self.counter = build_counter("test", prefer_exact=False)

    def _session(self, index: int, shared: float, target: int = 32000) -> Session:
        session = Session(index, self.corpus, self.counter, target, shared,
                          random.Random(index))
        session.reset()
        return session

    def _shared_tokens(self, a: Session, b: Session) -> int:
        first = [json.dumps(m, sort_keys=True) for m in a.messages]
        second = [json.dumps(m, sort_keys=True) for m in b.messages]
        common = 0
        for left, right in zip(first, second):
            if left != right:
                break
            common += 1
        return self.counter.count_messages(a.messages[:common])

    def test_shared_fraction_controls_the_shared_prefix(self):
        for target in (8000, 32000, 100000):
            measured = {}
            for fraction in (0.0, 0.5, 0.9):
                a, b = self._session(0, fraction, target), self._session(1, fraction, target)
                measured[fraction] = self._shared_tokens(a, b) / max(a.initial_tokens, 1)
            self.assertLess(measured[0.0], 0.12, f"target {target}")
            self.assertGreater(measured[0.9], measured[0.5], f"target {target}")
            self.assertGreater(measured[0.5], measured[0.0], f"target {target}")
            self.assertAlmostEqual(measured[0.9], 0.9, delta=0.12, msg=f"target {target}")

    def test_context_reaches_the_target(self):
        for target in (8000, 32000, 64000, 100000):
            session = self._session(0, 0.5, target)
            self.assertEqual(session.context_shortfall, 0, f"target {target}")
            self.assertGreater(session.initial_tokens, target * 0.5)
            self.assertLess(session.initial_tokens, target * 0.85)

    def test_steps_grow_the_context_and_compaction_bounds_it(self):
        session = self._session(0, 0.5, 8000)
        for _ in range(12):
            session.start_burst()
            for _ in range(6):
                session.next_step()
                session.record_step("done")
        actual = self.counter.count_messages(session.messages)
        self.assertGreater(session.compactions, 0, "long sessions must compact")
        self.assertLess(actual, 8000 * 2.0, "compaction must bound the context")

    def test_compaction_keeps_the_conversation_valid(self):
        """A tool message without the assistant turn that called it is an
        invalid conversation and some servers reject it outright."""
        session = self._session(0, 0.5, 8000)
        peak = 0
        for _ in range(20):
            session.start_burst()
            for _ in range(6):
                session.next_step()
                session.record_step("done")
            peak = max(peak, self.counter.count_messages(session.messages))
        self.assertGreater(session.compactions, 0)
        self.assertLess(peak, 8000 * 1.5)
        orphans = [
            index for index, message in enumerate(session.messages)
            if message.get("role") == "tool"
            and not (index and session.messages[index - 1].get("role") == "assistant"
                     and session.messages[index - 1].get("tool_calls"))
        ]
        self.assertEqual(orphans, [])

    def test_compaction_preserves_the_shared_prefix(self):
        a, b = self._session(0, 0.9, 8000), self._session(1, 0.9, 8000)
        before = self._shared_tokens(a, b)
        for session in (a, b):
            for _ in range(15):
                session.start_burst()
                for _ in range(5):
                    session.next_step()
                    session.record_step("done")
        self.assertGreater(a.compactions, 0)
        self.assertGreaterEqual(self._shared_tokens(a, b), before * 0.95)

    def test_text_style_produces_the_same_shape(self):
        session = Session(0, self.corpus, self.counter, 16000, 0.5,
                          random.Random(0), message_style="text")
        session.reset()
        self.assertTrue(all(m["role"] in ("system", "user", "assistant")
                            for m in session.messages))
        self.assertGreater(session.initial_tokens, 5000)


class TestPrometheus(unittest.TestCase):
    def test_parses_and_sums_label_sets(self):
        text = (
            "# HELP vllm:num_preemptions_total x\n"
            "# TYPE vllm:num_preemptions_total counter\n"
            'vllm:num_preemptions_total{model_name="q"} 12.0\n'
            'vllm:gpu_cache_usage_perc{model_name="q"} 0.73\n'
            'vllm:request_success_total{model_name="q",finished_reason="stop"} 4\n'
            'vllm:request_success_total{model_name="q",finished_reason="length"} 6\n'
        )
        totals = parse_prometheus(text)
        self.assertEqual(totals["vllm:num_preemptions_total"], 12.0)
        self.assertEqual(totals["vllm:request_success_total"], 10.0)
        self.assertEqual(pick(totals, "kv_cache_usage"), ("vllm:gpu_cache_usage_perc", 0.73))

    def test_v1_metric_names_are_recognised(self):
        totals = parse_prometheus("vllm:kv_cache_usage_perc 0.5\n"
                                  "vllm:prefix_cache_hits_total 90\n")
        self.assertEqual(pick(totals, "kv_cache_usage")[1], 0.5)
        self.assertEqual(pick(totals, "prefix_cache_hits")[1], 90.0)


class TestGrading(unittest.TestCase):
    def _aggregate(self, ttft, burst, decode, errors=0.0):
        return {"ttft": {"p90": ttft}, "burst_duration": {"p90": burst},
                "decode_tps": {"p50": decode}, "error_rate": errors}

    def test_green_run(self):
        grade = grade_run(self._aggregate(8.0, 40.0, 30.0),
                          {"preemptions": 0, "prefix_cache_hit_rate": 0.8})
        self.assertEqual(grade.colour, "groen")

    def test_preemptions_are_never_green(self):
        grade = grade_run(self._aggregate(3.0, 20.0, 40.0), {"preemptions": 12})
        self.assertEqual(grade.colour, "rood")

    def test_slow_instruction_beats_a_good_ttft(self):
        grade = grade_run(self._aggregate(6.0, 240.0, 30.0), {"preemptions": 0})
        self.assertEqual(grade.colour, "rood")

    def test_cache_hit_rate_warns_but_does_not_fail(self):
        grade = grade_run(self._aggregate(8.0, 40.0, 30.0),
                          {"preemptions": 0, "prefix_cache_hit_rate": 0.1})
        self.assertEqual(grade.colour, "groen")
        self.assertTrue(grade.warnings)

    def test_zero_shared_prefix_run_does_not_warn(self):
        grade = grade_run(self._aggregate(8.0, 40.0, 30.0),
                          {"preemptions": 0, "prefix_cache_hit_rate": 0.05},
                          expects_shared_prefix=False)
        self.assertFalse(grade.warnings)


class TestMatrix(unittest.TestCase):
    def test_default_config_builds_a_plan(self):
        config = load_jsonc(os.path.join(ROOT, "config", "default.json"))
        specs = build_specs(config)
        self.assertEqual(len({s.run_id for s in specs}), len(specs), "run ids must be unique")
        plan = describe_plan(specs, 30)
        self.assertGreater(plan["runs"], 30)
        self.assertLess(plan["total_seconds"], 8 * 3600, "phase 1 must stay under a workday")
        sweep = [s for s in specs if s.kind == "sweep"]
        self.assertEqual(len(sweep), 16)

    def test_smoke_config_is_short(self):
        config = load_jsonc(os.path.join(ROOT, "config", "smoke.json"))
        plan = describe_plan(build_specs(config), 3)
        self.assertLess(plan["total_seconds"], 40 * 60)

    def test_every_run_has_a_measured_phase(self):
        config = load_jsonc(os.path.join(ROOT, "config", "default.json"))
        for spec in build_specs(config) + build_specs(config, ["lesson"]):
            self.assertTrue(any(p.measure for p in spec.phases), spec.run_id)


class TestRampUp(unittest.TestCase):
    """The cliff finder produces the single most useful number in the whole
    test, and a crash in its controller is silent: the run simply finishes
    without a verdict. So it gets its own end-to-end test."""

    def test_controller_admits_students_and_reports_a_maximum(self):
        import asyncio
        from stresstest.client import Endpoint, OpenAIClient
        from stresstest.matrix import build_rampup
        from stresstest.mockserver import build_server
        from stresstest.runner import RunEngine
        from stresstest.vllm_metrics import MetricsSampler

        config = {
            "seed": 3, "request": {"ignore_eos": True},
            "run_defaults": {"drain_s": 8, "progress_interval_s": 60},
            "matrix": {"rampup": {"start_students": 2, "max_students": 5,
                                  "step_interval_s": 3, "context_tokens": 4000,
                                  "activity": "intensief"}},
        }
        spec = build_rampup(config)[0]
        self.assertNotIn(None, list(spec.ramp.values()),
                         "a None limit would crash the controller on float()")

        server = build_server("127.0.0.1", 8766, gpu_memory_gb=8, model_gb=4,
                              kv_bytes_per_token=200_000, prefill_tokens_per_s=20000,
                              decode_tokens_per_s=400, max_num_seqs=8,
                              max_model_len=131072)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.3)
        try:
            endpoint = Endpoint(base_url="http://127.0.0.1:8766/v1",
                                model="mock-model", timeout_s=60)
            engine = RunEngine(client=OpenAIClient(endpoint), corpus=make_corpus(),
                               counter=build_counter("t", prefer_exact=False),
                               config=config)
            sampler = MetricsSampler(None)
            result = asyncio.run(engine.run(spec, sampler, progress=False))
        finally:
            server.shutdown()
            server.server_close()

        self.assertIsNotNone(result.ramp, "the controller must produce a verdict")
        self.assertGreaterEqual(result.ramp["max_students_ok"], 2)
        self.assertTrue(result.ramp["steps"], "every step must be recorded")
        seen = [step["students"] for step in result.ramp["steps"]]
        self.assertEqual(seen, sorted(seen), "students are admitted one at a time")
        active = {record.student_index for record in result.collector.requests}
        self.assertLessEqual(max(active) + 1, 5,
                             "no student beyond the admitted count may run")


class TestEndToEnd(unittest.TestCase):
    """One tiny run against the built-in mock, exercising client, runner,
    metrics, grading and the report writer together."""

    def test_full_pipeline(self):
        import asyncio
        from stresstest.client import Endpoint, OpenAIClient
        from stresstest.mockserver import build_server
        from stresstest.report import ResultsWriter, write_analysis
        from stresstest.runner import RunEngine
        from stresstest.runspec import standard_phases, RunSpec
        from stresstest.vllm_metrics import MetricsSampler
        from stresstest import resultaten
        import tempfile

        server = build_server("127.0.0.1", 8765, gpu_memory_gb=8, model_gb=4,
                              kv_bytes_per_token=200_000, prefill_tokens_per_s=20000,
                              decode_tokens_per_s=400, max_num_seqs=8,
                              max_model_len=131072)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.3)
        try:
            config = {
                "seed": 1, "request": {"ignore_eos": True},
                "run_defaults": {"drain_s": 10, "progress_interval_s": 30},
                "hardware": {"vram_gb": 8, "gpu_memory_utilization": 0.9,
                             "model_weights_gb": 4,
                             "alternatives": [{"name": "kleiner", "vram_gb": 6,
                                               "price_eur": 100}]},
                "output": {"directory": "x"},
            }
            endpoint = Endpoint(base_url="http://127.0.0.1:8765/v1",
                                model="mock-model", timeout_s=60)
            engine = RunEngine(client=OpenAIClient(endpoint), corpus=make_corpus(),
                               counter=build_counter("t", prefer_exact=False),
                               config=config)
            spec = RunSpec(run_id="sweep_s4_c8k", label="test", kind="sweep",
                           students=4, context_tokens=8000, shared_fraction=0.5,
                           phases=standard_phases(4, 12), arrival_window_s=1.0)
            sampler = MetricsSampler("http://127.0.0.1:8765/metrics", interval_s=1.0)
            result = asyncio.run(engine.run(spec, sampler, progress=False))

            self.assertGreater(result.aggregate["requests_total"], 0)
            self.assertEqual(result.aggregate["requests_failed"], 0)
            self.assertTrue(result.server["metrics_available"])
            self.assertIn("prefix_cache_hit_rate", result.server)
            self.assertIn(result.grade.colour, ("groen", "oranje", "rood"))

            with tempfile.TemporaryDirectory() as directory:
                writer = ResultsWriter(directory, config, {"version": "test"})
                writer.add(result)
                charts = writer.write_charts()
                self.assertTrue(charts)
                findings = write_analysis(directory, writer.results, config)
                self.assertIn("alternatives", findings)
                path = resultaten.write(directory, writer.results, config,
                                        {"version": "test", "model": "mock"})
                with open(path, encoding="utf-8") as handle:
                    report = handle.read()
                self.assertIn("Resultaten stresstest", report)
                self.assertIn("Waar ligt de klif", report)
                self.assertTrue(os.path.exists(
                    os.path.join(directory, "runs", "sweep_s4_c8k", "requests.csv")))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
