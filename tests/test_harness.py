"""Tests for the parts where a silent mistake would invalidate a whole run.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stresstest.conversation import Session
from stresstest.corpus import CodeCorpus, CorpusGroup, SourceFile
from stresstest.grading import grade_run
from stresstest.matrix import build_specs, describe_plan
from stresstest.personas import (DEFAULT_PERSONAS, DEFAULT_WORK_PROFILES, WorkProfile,
                                 build_class, class_composition, work_composition)
from stresstest.tokens import build_counter
from stresstest.util import load_jsonc, percentile, sample_lognormal
from stresstest.vllm_metrics import parse_prometheus, pick

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def real_corpus() -> CodeCorpus:
    """The harness's own source: real code, always present, no clone needed.

    Token density is corpus dependent -- docstring-heavy Python sits around
    4.0 characters per token, PHP and JavaScript nearer 4.8 -- so a test that
    pins the ratio must use real source rather than the repetitive synthetic
    files used elsewhere.
    """
    from stresstest.corpus import _iter_source_files
    files = list(_iter_source_files(os.path.join(ROOT, "stresstest"), [".py"]))
    if not files:
        raise unittest.SkipTest("eigen broncode niet gevonden")
    return CodeCorpus("self", files, max(1, len(files) // 2))


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


class TestWorkProfiles(unittest.TestCase):
    """The two behaviour axes. A persona is a pace, a work profile is a
    weight, and the whole point of splitting them is that they mix freely."""

    def test_both_axes_are_allocated_and_are_not_correlated(self):
        profiles = build_class(20, DEFAULT_PERSONAS, "normaal", 32000, 20250908)
        self.assertEqual(sum(class_composition(profiles).values()), 20)
        self.assertEqual(sum(work_composition(profiles).values()), 20)
        # Every work profile in the default mix must actually appear, and the
        # heavy one must not land on a single persona.
        work = work_composition(profiles)
        self.assertEqual(set(work), {"klein", "middel", "doorspitten"})
        heavy_personas = {p.persona.name for p in profiles if p.work.name == "doorspitten"}
        self.assertGreater(len(heavy_personas), 1,
                           "the heavy profile landed on one persona only; "
                           "the axes are correlated")

    def test_the_class_is_reproducible_on_both_axes(self):
        a = build_class(20, DEFAULT_PERSONAS, "normaal", 32000, 7)
        b = build_class(20, DEFAULT_PERSONAS, "normaal", 32000, 7)
        self.assertEqual([(x.persona.name, x.work.name) for x in a],
                         [(x.persona.name, x.work.name) for x in b])

    def test_a_heavier_profile_produces_heavier_steps(self):
        """The reason the profiles exist. If 'doorspitten' does not actually
        return more per tool call and emit more per step, the matrix measures
        the same thing three times."""
        corpus = make_corpus(files=80, shared=20)
        counter = build_counter("test", prefer_exact=False)
        sizes = {}
        for work in DEFAULT_WORK_PROFILES:
            session = Session(0, corpus, counter, 32000, 0.5, random.Random(3),
                              work=work)
            session.reset()
            results, outputs = [], []
            for _ in range(400):
                _, arguments, result = session._synthesise_tool_result()
                results.append(counter.count(result) + counter.count(arguments))
                outputs.append(session._step_output_tokens())
            results.sort(); outputs.sort()
            sizes[work.name] = (results[len(results) // 2], outputs[len(outputs) // 2])
        light, heavy = sizes["klein"], sizes["doorspitten"]
        self.assertGreater(heavy[0], 3 * light[0],
                           f"tool results per profile: {sizes}")
        self.assertGreater(heavy[1], 2 * light[1],
                           f"model output per profile: {sizes}")


class TestCorpusGroups(unittest.TestCase):
    """A group is one assignment. Students inside one share a prefix; students
    in different ones must not, or the cache axis measures the wrong thing."""

    def _corpus(self) -> CodeCorpus:
        def files(prefix, n, body):
            return [SourceFile(f"{prefix}/f{i:03d}.cs", body * (i % 5 + 3))
                    for i in range(n)]
        return CodeCorpus("two", [
            CorpusGroup("web", files("web", 40, "def a():\n    return 1\n" * 20), 15),
            CorpusGroup("unity", files("unity", 40, "void Update() { }\n" * 60), 15),
        ])

    def test_a_flat_corpus_still_behaves_as_one_group(self):
        corpus = make_corpus(files=30, shared=10)
        self.assertEqual(len(corpus.groups), 1)
        self.assertEqual(len(corpus.shared_files), 10)
        self.assertEqual(corpus.student_files(0, 5), corpus.group().student_files(0, 5))

    def test_groups_do_not_share_files(self):
        corpus = self._corpus()
        web = {f.path for f in corpus.group("web").files}
        unity = {f.path for f in corpus.group("unity").files}
        self.assertFalse(web & unity)

    def test_two_students_in_different_groups_share_no_project_prefix(self):
        corpus = self._corpus()
        counter = build_counter("test", prefer_exact=False)

        def session(group):
            work = WorkProfile(name=group, share=1.0, group=group)
            s = Session(0, corpus, counter, 16000, 0.9, random.Random(1), work=work)
            s.reset()
            return s

        a, b = session("web"), session("unity")
        common = 0
        for left, right in zip(a.messages, b.messages):
            if json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
                break
            common += 1
        # The system prompt and the assignment brief are shared by everyone;
        # anything beyond that would mean the groups leaked into each other.
        self.assertLessEqual(common, 2, "groups must not share project files")

    def test_an_unknown_group_fails_loudly(self):
        with self.assertRaises(RuntimeError):
            self._corpus().group("bestaat-niet")


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

    def test_compaction_is_counted_so_the_results_can_report_it(self):
        """How often a session hits its context ceiling is the empirical
        answer to 'is this context size big enough'. The agent already had to
        compact; if the count never leaves the Session, every run has to
        reconstruct it from prompt-token traces afterwards."""
        session = self._session(0, 0.5, target=8000)
        self.assertEqual(session.compactions, 0)
        # Drive it well past its target: several instructions of a dozen steps.
        for _ in range(12):
            session.start_burst()
            for _ in range(12):
                session.next_step()
                session.record_step("ok")
        self.assertGreater(session.compactions, 0,
                           "a session driven past its target must compact")
        self.assertLessEqual(session.prompt_tokens_estimate, session.target_tokens * 1.6,
                             "compaction has to actually bring the context back down")

    def test_a_roomy_context_compacts_far_less_often(self):
        """The other side of the same measurement. A big window does not
        abolish compaction -- run long enough and every window fills -- but it
        pushes it out, and that ratio is what the context axis is asking
        about."""
        counts = {}
        for target in (8000, 100000):
            session = self._session(0, 0.5, target=target)
            for _ in range(12):
                session.start_burst()
                for _ in range(12):
                    session.next_step()
                    session.record_step("ok")
            counts[target] = session.compactions
        self.assertGreater(counts[8000], 3 * counts[100000],
                           f"compactions per target: {counts}")

    def test_context_reaches_the_target_without_overshooting(self):
        """The initial context fills to roughly 70 % of the target -- the rest
        is headroom for the live agent steps -- and never past the target
        itself, because the context size is a column of the matrix."""
        for target in (8000, 32000, 64000, 100000):
            session = self._session(0, 0.5, target)
            self.assertEqual(session.context_shortfall, 0, f"target {target}")
            self.assertGreater(session.initial_tokens, target * 0.5, f"target {target}")
            self.assertLessEqual(session.initial_tokens, target, f"target {target}")

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


class TestTokenEstimate(unittest.TestCase):
    """The context size is an axis of the whole matrix. If the fallback
    estimate is systematically wrong, every conclusion about memory is wrong
    with it -- which is exactly what a 3.5 characters-per-token rule of thumb
    borrowed from English prose did before this was measured."""

    MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

    def _exact(self):
        counter = build_counter(self.MODEL, prefer_exact=True)
        if not counter.exact:
            self.skipTest("geen exacte tokenizer beschikbaar "
                          "(pip install tokenizers, plus netwerktoegang)")
        return counter

    def test_default_ratio_matches_the_real_tokenizer(self):
        from stresstest.conversation import TOOLS
        from stresstest.tokens import DEFAULT_CHARS_PER_TOKEN, TokenCounter

        exact = self._exact()
        corpus = real_corpus()
        estimate = TokenCounter("estimate", "test", DEFAULT_CHARS_PER_TOKEN)
        worst = 0.0
        for target in (8000, 32000, 100000):
            session = Session(0, corpus, estimate, target, 0.5, random.Random(0))
            session.reset()
            measured = exact.count_messages(session.messages) + exact.count_tools(TOOLS)
            worst = max(worst, abs(session.initial_tokens - measured) / measured)
        # 20 % is deliberately loose: the ratio depends on the corpus, which
        # is why `stresstest calibrate` exists. What must not happen again is
        # the 32 % the prose rule of thumb produced.
        self.assertLess(worst, 0.20,
                        f"schatting wijkt {worst:.0%} af van de echte tokenizer")

    def test_exact_counter_still_reaches_the_target(self):
        exact = self._exact()
        corpus = real_corpus()
        for target in (8000, 32000, 64000):
            session = Session(0, corpus, exact, target, 0.5, random.Random(0))
            session.reset()
            self.assertEqual(session.context_shortfall, 0, f"doel {target}")
            self.assertGreater(session.initial_tokens, target * 0.5, f"doel {target}")
            self.assertLessEqual(session.initial_tokens, target, f"doel {target}")

    def test_solver_finds_a_ratio_that_beats_the_prose_rule_of_thumb(self):
        from stresstest.conversation import TOOLS
        from stresstest.tokens import TokenCounter, best_ratio

        exact = self._exact()
        corpus = real_corpus()

        def errors_for(ratio: float) -> list[float]:
            estimate = TokenCounter("estimate", "test", ratio)
            out = []
            for target in (8000, 32000):
                session = Session(0, corpus, estimate, target, 0.5, random.Random(0))
                session.reset()
                measured = (exact.count_messages(session.messages)
                            + exact.count_tools(TOOLS))
                out.append((session.initial_tokens - measured) / measured)
            return out

        ratio, worst = best_ratio(errors_for)
        self.assertGreater(ratio, 3.9, "code is niet zo dicht als proza")
        self.assertLess(worst, max(abs(e) for e in errors_for(3.5)))


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


class TestMonitor(unittest.TestCase):
    """`monitor` watches a server somebody else is loading. Two things have to
    hold: the time series must land on disk as it arrives (the pod gets killed,
    the SSH session drops, the lesson ends abruptly), and the file must have the
    same shape as the one a run writes, or the real lesson cannot be plotted
    next to the simulated one."""

    def _serve(self, pages):
        """A fake /metrics that hands out `pages` in order, then repeats the last."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        state = {"i": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = pages[min(state["i"], len(pages) - 1)].encode()
                state["i"] += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # silence
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/metrics"

    @staticmethod
    def _page(queries, hits, usage, running, waiting, preemptions=0):
        return (f"vllm:prefix_cache_queries_total {queries}\n"
                f"vllm:prefix_cache_hits_total {hits}\n"
                f"vllm:kv_cache_usage_perc {usage}\n"
                f"vllm:num_requests_running {running}\n"
                f"vllm:num_requests_waiting {waiting}\n"
                f"vllm:num_preemptions_total {preemptions}\n"
                f"vllm:prompt_tokens_total {queries * 10}\n"
                f"vllm:generation_tokens_total {queries}\n")

    def test_writes_the_same_columns_a_run_writes(self):
        import csv as csv_module
        import tempfile
        from stresstest.cli import MONITOR_COLUMNS, build_parser, cmd_monitor

        url = self._serve([self._page(50, 40, 0.05, 1, 0),
                           self._page(100, 90, 0.10, 2, 0),
                           self._page(200, 190, 0.25, 5, 1),
                           self._page(300, 285, 0.20, 3, 0)])
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(
                ["monitor", "--url", url, "--out", tmp, "--interval", "0.05",
                 "--duration", "0.6", "--status-interval", "10"])
            self.assertEqual(cmd_monitor(args), 0)
            with open(os.path.join(tmp, "server_metrics.csv"), encoding="utf-8") as handle:
                rows = list(csv_module.DictReader(handle))

        self.assertGreaterEqual(len(rows), 3, "de tijdreeks moet regels bevatten")
        self.assertEqual(list(rows[0].keys()), MONITOR_COLUMNS)
        # Everything a run's server_metrics.csv carries must be here too, or the
        # two cannot go on the same axes.
        for column in ("t_s", "t_wall", "kv_cache_usage", "requests_running",
                       "requests_waiting", "preemptions", "prefix_cache_queries",
                       "prefix_cache_hits", "prefix_cache_hit_rate_window"):
            self.assertIn(column, rows[0])
        self.assertAlmostEqual(float(rows[0]["t_s"]), 0.0, places=2)

    def test_the_summary_reports_what_the_server_did(self):
        import json as json_module
        import tempfile
        from stresstest.cli import build_parser, cmd_monitor

        # check() consumes the first page, so the two that carry the assertions
        # are the second and third; the fake repeats the last one after that.
        url = self._serve([self._page(50, 40, 0.05, 1, 0),
                           self._page(100, 90, 0.10, 2, 0),
                           self._page(200, 190, 0.50, 9, 4, preemptions=3)])
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(
                ["monitor", "--url", url, "--out", tmp, "--interval", "0.05",
                 "--duration", "0.5", "--status-interval", "10",
                 "--set", "hardware.vram_gb=96",
                 "--set", "hardware.gpu_memory_utilization=0.9",
                 "--set", "hardware.model_weights_gb=31.1"])
            self.assertEqual(cmd_monitor(args), 0)
            payload = json_module.load(open(os.path.join(tmp, "monitor.json"),
                                            encoding="utf-8"))

        server = payload["server"]
        self.assertEqual(payload["kind"], "monitor")
        self.assertAlmostEqual(payload["kv_pool_gb"], 55.3, places=1)
        self.assertAlmostEqual(server["kv_cache_usage_peak"], 0.50, places=6)
        self.assertEqual(server["queue_depth_peak"], 4.0)
        self.assertEqual(server["requests_running_peak"], 9.0)
        # 100 extra queries, 100 extra hits over the window.
        self.assertAlmostEqual(server["prefix_cache_hit_rate"], 1.0, places=6)
        self.assertEqual(server["preemptions"], 3.0)
        # The peak in gigabytes is what makes it comparable to the reports.
        self.assertAlmostEqual(payload["kv_peak_gb"], 0.50 * payload["kv_pool_gb"],
                               places=1)

    def test_every_sample_is_on_disk_before_the_next_one(self):
        """The property that matters when the pod is killed mid-lesson."""
        import csv as csv_module
        import tempfile
        from stresstest.cli import _MetricsCsv
        from stresstest.vllm_metrics import MetricSample

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "server_metrics.csv")
            writer = _MetricsCsv(path)
            for index in range(3):
                writer.add(MetricSample(t_monotonic=100.0 + index, t_wall=1.0 + index,
                                        values={"kv_cache_usage": 0.1 * index,
                                                "prefix_cache_queries": 10.0 * (index + 1),
                                                "prefix_cache_hits": 8.0 * (index + 1)}))
                # Read it back with the writer still open and unflushed-by-us.
                with open(path, encoding="utf-8") as handle:
                    rows = list(csv_module.DictReader(handle))
                self.assertEqual(len(rows), index + 1,
                                 "elk monster moet meteen op schijf staan")
            self.assertAlmostEqual(float(rows[-1]["t_s"]), 2.0, places=2)
            self.assertAlmostEqual(float(rows[-1]["prefix_cache_hit_rate_window"]),
                                   0.8, places=6)
            writer.close()


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


class TestFingerprint(unittest.TestCase):
    """A run id says students and context and nothing else. Everything the id
    leaves out -- corpus, behaviour model, tokenizer, attention backend -- is
    what the fingerprint carries, so a resume can tell "already measured" from
    "measured with something else"."""

    def setUp(self):
        import tempfile
        from stresstest.cli import load_config
        self.config = load_config(os.path.join(ROOT, "config", "default.json"))
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _store(self, run_id: str, mark) -> None:
        run_dir = os.path.join(self.tmp, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump({"spec": {"run_id": run_id}, "fingerprint": mark}, handle)

    def test_the_same_configuration_gives_the_same_fingerprint(self):
        from stresstest import fingerprint
        first = fingerprint.from_config(self.config)
        second = fingerprint.from_config(json.loads(json.dumps(self.config)))
        self.assertEqual(fingerprint.digest(first), fingerprint.digest(second))
        self.assertEqual(fingerprint.differences(first, second), [])

    def test_a_changed_behaviour_model_is_a_difference(self):
        """The reproduction from the issue: change a work profile and the run
        ids stay identical, so only the fingerprint can notice."""
        from stresstest import fingerprint
        before = fingerprint.from_config(self.config)
        changed = json.loads(json.dumps(self.config))
        changed["behaviour"]["work_profiles"][0]["share"] = 0.9
        after = fingerprint.from_config(changed)
        paths = [path for path, _, _ in fingerprint.differences(before, after)]
        self.assertIn("behaviour.work_profiles.klein.share", paths)
        self.assertNotEqual(fingerprint.digest(before), fingerprint.digest(after))

    def test_corpus_tokenizer_and_backend_all_count(self):
        from stresstest import fingerprint
        before = fingerprint.from_config(self.config)
        for path, value in ((("corpus", "extensions"), ".py"),
                            (("tokenizer", "prefer_exact"), False),
                            (("hardware", "attention_backend"), "TRITON_ATTN"),
                            (("seed",), 999)):
            changed = json.loads(json.dumps(self.config))
            node = changed
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            self.assertTrue(fingerprint.differences(before, fingerprint.from_config(changed)),
                            f"{'.'.join(path)} veranderen zou een verschil moeten zijn")

    def test_a_field_only_one_side_knows_is_not_a_difference(self):
        """The cheap check knows only the configuration; a full run also knows
        the corpus file counts. They share one stored fingerprint, so absence
        on one side must not read as disagreement."""
        from stresstest import fingerprint
        cheap = fingerprint.from_config(self.config)
        full = dict(cheap, corpus_files={"name": "web", "files": 120},
                    tokenizer_exact=True)
        self.assertEqual(fingerprint.differences(full, cheap), [])

    def test_the_path_to_the_tokenizer_is_recorded_but_not_compared(self):
        """Two pods keep the model in different directories. That is not a
        different measurement, and stopping a six-hour run over it would be."""
        from stresstest import fingerprint
        here = json.loads(json.dumps(self.config))
        here["tokenizer"]["path"] = "/workspace/hf/snapshot-a"
        there = json.loads(json.dumps(self.config))
        there["tokenizer"]["path"] = "/root/hf/snapshot-a"
        left, right = fingerprint.from_config(here), fingerprint.from_config(there)
        self.assertEqual(left["tokenizer"]["path"], "/workspace/hf/snapshot-a")
        self.assertEqual(fingerprint.differences(left, right), [])
        self.assertEqual(fingerprint.digest(left), fingerprint.digest(right))

    def test_the_guard_refuses_a_directory_measured_with_another_setup(self):
        from stresstest import fingerprint
        changed = json.loads(json.dumps(self.config))
        changed["behaviour"]["personas"][0]["think_time_s"] = [1, 2]
        self._store("sweep_s20_c8k", fingerprint.from_config(changed))
        with self.assertRaises(fingerprint.Mismatch) as caught:
            fingerprint.guard(self.tmp, fingerprint.from_config(self.config))
        message = caught.exception.message()
        self.assertIn("sweep_s20_c8k", message)
        self.assertIn("think_time_s", message)
        self.assertIn("RESULTS_DIR=", message, "de melding moet de uitweg noemen")
        self.assertIn("--resume-anyway", message)

    def test_the_guard_lets_a_matching_directory_through(self):
        from stresstest import fingerprint
        mark = fingerprint.from_config(self.config)
        self._store("sweep_s20_c8k", mark)
        self.assertEqual(fingerprint.guard(self.tmp, mark), {})

    def test_resume_anyway_reports_instead_of_stopping(self):
        from stresstest import fingerprint
        changed = json.loads(json.dumps(self.config))
        changed["seed"] = 7
        self._store("sweep_s20_c8k", fingerprint.from_config(changed))
        said = []
        offenders = fingerprint.guard(self.tmp, fingerprint.from_config(self.config),
                                      allow_mismatch=True, warn=said.append)
        self.assertIn("sweep_s20_c8k", offenders)
        self.assertTrue(any("seed" in line for line in said),
                        "toch hervatten mag, stilzwijgend niet")

    def test_runs_without_a_fingerprint_are_reported_not_accused(self):
        """Everything measured before this existed. Unknown is not the same as
        wrong, and neither one is the same as fine."""
        from stresstest import fingerprint
        run_dir = os.path.join(self.tmp, "runs", "sweep_s5_c8k")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump({"spec": {"run_id": "sweep_s5_c8k"}}, handle)
        said = []
        self.assertEqual(
            fingerprint.guard(self.tmp, fingerprint.from_config(self.config),
                              warn=said.append), {})
        self.assertTrue(any("vingerafdruk" in line for line in said))

    def test_engine_variants_are_not_refused_for_varying_the_engine(self):
        """`engine_kv_fp8` restarts vLLM with another KV dtype: that is what it
        measures. A check that called it a changed setup would stop the group
        it is meant to protect."""
        from stresstest import fingerprint
        baseline = json.loads(json.dumps(self.config))
        baseline["hardware"]["kv_cache_dtype"] = "auto"
        self._store("sweep_s20_c8k", fingerprint.from_config(baseline))
        variant = json.loads(json.dumps(self.config))
        variant["hardware"]["kv_cache_dtype"] = "fp8"
        current = fingerprint.from_config(variant)
        with self.assertRaises(fingerprint.Mismatch):
            fingerprint.guard(self.tmp, current)
        self.assertEqual(fingerprint.guard(self.tmp, current,
                                           adding_engine_runs=True), {})

    def test_a_stored_engine_run_does_not_block_the_next_group(self):
        from stresstest import fingerprint
        variant = json.loads(json.dumps(self.config))
        variant["hardware"]["kv_cache_dtype"] = "fp8"
        run_dir = os.path.join(self.tmp, "runs", "engine_kv_fp8")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump({"spec": {"run_id": "engine_kv_fp8", "kind": "engine"},
                       "fingerprint": fingerprint.from_config(variant)}, handle)
        self.assertEqual(fingerprint.guard(self.tmp,
                                           fingerprint.from_config(self.config)), {})

    def test_an_engine_run_is_still_held_to_the_rest_of_the_setup(self):
        """Only the engine settings get the exception. A different corpus is a
        different measurement whatever the run is called."""
        from stresstest import fingerprint
        changed = json.loads(json.dumps(self.config))
        changed["corpus"]["extensions"] = ".py"
        self._store("sweep_s20_c8k", fingerprint.from_config(changed))
        with self.assertRaises(fingerprint.Mismatch):
            fingerprint.guard(self.tmp, fingerprint.from_config(self.config),
                              adding_engine_runs=True)

    def test_the_digest_survives_the_engine_variants(self):
        """One matrix directory, one digest -- otherwise `fingerprint --same`
        would say a matrix and its own lesson disagree."""
        from stresstest import fingerprint
        auto = json.loads(json.dumps(self.config))
        auto["hardware"]["kv_cache_dtype"] = "auto"
        fp8 = json.loads(json.dumps(self.config))
        fp8["hardware"]["kv_cache_dtype"] = "fp8"
        self.assertEqual(fingerprint.digest(fingerprint.from_config(auto)),
                         fingerprint.digest(fingerprint.from_config(fp8)))
        changed = json.loads(json.dumps(self.config))
        changed["seed"] = 3
        self.assertNotEqual(fingerprint.digest(fingerprint.from_config(auto)),
                            fingerprint.digest(fingerprint.from_config(changed)))

    def test_pending_stops_on_a_mismatch_and_lists_ids_otherwise(self):
        """What scripts/pod.sh calls before every group."""
        from stresstest import fingerprint
        self._store("sweep_s5_c8k", fingerprint.from_config(self.config))
        command = [sys.executable, "-m", "stresstest", "pending", self.tmp,
                   "--only", "sweep", "-c",
                   os.path.join(ROOT, "config", "default.json")]
        done = subprocess.run(command, capture_output=True, text=True, cwd=ROOT,
                              timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("sweep_s5_c8k", done.stdout.split(),
                         "een run die er al staat is niet pending")
        self.assertIn("sweep_s20_c8k", done.stdout.split())

        done = subprocess.run(command + ["--set", "seed=4321"], capture_output=True,
                              text=True, cwd=ROOT, timeout=120)
        self.assertEqual(done.returncode, 3, done.stdout)
        self.assertIn("seed", done.stderr)
        self.assertEqual(done.stdout.strip(), "",
                         "bij een afwijking mag er geen id-lijst uitkomen")

        done = subprocess.run(command + ["--set", "seed=4321", "--resume-anyway"],
                              capture_output=True, text=True, cwd=ROOT, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("sweep_s20_c8k", done.stdout.split())

    def test_the_directory_fingerprint_is_readable_from_the_command_line(self):
        """How pod.sh decides whether an older lesson belongs with this matrix."""
        from stresstest import fingerprint
        other = os.path.join(self.tmp, "elders")
        os.makedirs(other)
        mark = fingerprint.from_config(self.config)
        for directory, payload in ((self.tmp, mark),
                                   (other, fingerprint.from_config(
                                       dict(self.config, seed=11)))):
            with open(os.path.join(directory, "environment.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"fingerprint": payload,
                           "fingerprint_digest": fingerprint.digest(payload)}, handle)
        command = [sys.executable, "-m", "stresstest", "fingerprint", "--same"]
        self.assertEqual(subprocess.run(command + [self.tmp, self.tmp], cwd=ROOT,
                                        capture_output=True, timeout=120).returncode, 0)
        self.assertEqual(subprocess.run(command + [self.tmp, other], cwd=ROOT,
                                        capture_output=True, timeout=120).returncode, 1)


class TestReportProvenance(unittest.TestCase):
    """RESULTATEN.md is the file the budget request rests on. It has to say
    which measurements it is made of, and say it loudly when they disagree."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _directory(self, name: str, seed: int, day: str, runs=("sweep_s5_c8k",)):
        from stresstest import fingerprint
        from stresstest.cli import load_config
        config = load_config(os.path.join(ROOT, "config", "default.json"))
        config["seed"] = seed
        mark = fingerprint.from_config(config)
        directory = os.path.join(self.tmp, name)
        os.makedirs(os.path.join(directory, "runs"), exist_ok=True)
        with open(os.path.join(directory, "environment.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"fingerprint": mark,
                       "fingerprint_digest": fingerprint.digest(mark),
                       "generated": "2026-01-01T00:00:00Z"}, handle)
        for run_id in runs:
            run_dir = os.path.join(directory, "runs", run_id)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
                json.dump({"spec": {"run_id": run_id},
                           "started": f"{day}T09:00:00Z", "fingerprint": mark}, handle)
        return directory

    def test_a_source_is_dated_by_its_runs_not_by_today(self):
        """environment.json is rewritten every time the report is regenerated,
        so its timestamp is a rendering date. The runs know better."""
        from stresstest.report import describe_source
        described = describe_source(self._directory("een", 1, "2026-09-08"))
        self.assertEqual(described["measured"], "2026-09-08")
        self.assertEqual(described["runs"], 1)
        self.assertTrue(described["fingerprint"])

    def test_two_measurements_in_one_report_are_named_and_flagged(self):
        from stresstest import resultaten
        first = self._directory("matrix", 1, "2026-09-08")
        second = self._directory("les", 2, "2026-09-09")
        text = resultaten.render([], {}, {}, sources=[first, second])
        self.assertIn("2026-09-08", text)
        self.assertIn("2026-09-09", text)
        self.assertIn("niet met dezelfde opstelling gemaakt", text,
                      "twee gedragsmodellen in een rapport horen bovenaan te staan")

    def test_one_measurement_is_not_flagged(self):
        from stresstest import resultaten
        first = self._directory("matrix", 1, "2026-09-08")
        second = self._directory("les", 1, "2026-09-08")
        text = resultaten.render([], {}, {}, sources=[first, second])
        self.assertNotIn("niet met dezelfde opstelling gemaakt", text)
        self.assertIn(first, text, "de bronnen horen er nog wel bij te staan")
        self.assertIn(second, text)

    def test_a_source_without_a_fingerprint_says_so(self):
        from stresstest import resultaten
        known = self._directory("matrix", 1, "2026-09-08")
        older = os.path.join(self.tmp, "oud")
        os.makedirs(os.path.join(older, "runs"))
        text = resultaten.render([], {}, {}, sources=[known, older])
        self.assertIn("niet vast te stellen", text)


class TestBehaviourModelInTheReport(unittest.TestCase):
    """Who the class was drawn from, and who it turned out to be, are two
    claims. Reporting the first run's draw as "the class" answers neither."""

    def _result(self, run_id, kind, students, composition):
        from stresstest.grading import Grade
        from stresstest.metrics import Collector
        from stresstest.runner import RunResult
        from stresstest.runspec import RunSpec
        spec = RunSpec(run_id=run_id, label=run_id, kind=kind, students=students,
                       context_tokens=8000, phases=[])
        grade = Grade(colour="groen", reasons=[], warnings=[])
        return RunResult(spec=spec, started_wall=0.0, finished_wall=1.0,
                         aggregate={}, server={}, grade=grade, grade_brief=grade,
                         composition=composition, collector=Collector(run_id),
                         metric_rows=[])

    def _config(self):
        from stresstest.cli import load_config
        return load_config(os.path.join(ROOT, "config", "default.json"))

    def test_a_smaller_run_is_never_presented_as_the_class(self):
        """The bug: with only sweep runs present, the ten-student draw was
        published under "who the class consisted of"."""
        from stresstest.report import analyse
        findings = analyse([self._result("sweep_s10_c8k", "sweep", 10,
                                         {"gemiddelde": 5, "afhaker": 5})],
                           self._config())
        self.assertIsNone(findings.get("class_mix"),
                          "een run van tien studenten beschrijft de klas niet")
        self.assertEqual(findings["behaviour_model"]["personas"]["gemiddelde"], 0.45)

    def test_the_class_run_is_used_and_named(self):
        from stresstest.report import analyse
        config = self._config()
        findings = analyse([self._result("sweep_s10_c8k", "sweep", 10, {"afhaker": 10}),
                            self._result("les_90min", "lesson", 20,
                                         {"gemiddelde": 9, "werk": {"klein": 10}})],
                           config)
        self.assertEqual(findings["class_mix"]["run_id"], "les_90min")
        self.assertEqual(findings["class_mix"]["students"], 20)
        self.assertEqual(findings["class_mix"]["personas"], {"gemiddelde": 9})

    def test_an_activity_run_does_not_stand_in_for_the_class(self):
        """`act_intensief_s20` has twenty students and a deliberately unusual
        mix -- that is what it is for."""
        from stresstest.report import analyse
        findings = analyse([self._result("act_intensief_s20", "activity", 20,
                                         {"doorpakker": 20})], self._config())
        self.assertIsNone(findings.get("class_mix"))

    def test_the_behaviour_model_travels_with_the_report(self):
        from stresstest import resultaten
        from stresstest.report import analyse
        config = self._config()
        results = [self._result("les_90min", "lesson", 20,
                                {"gemiddelde": 9, "werk": {"klein": 10}})]
        text = resultaten.render(results, config, {"model": "mock"})
        self.assertIn("les_90min", text, "de bron van de loting hoort in het rapport")
        self.assertIn(str(config["seed"]), text)
        self.assertIn(analyse(results, config)["behaviour_model"]["fingerprint"], text)


if __name__ == "__main__":
    unittest.main()
