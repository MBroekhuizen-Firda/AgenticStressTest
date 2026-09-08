"""Tests for scripts/pod.sh, the unattended driver.

The wrapper drives seven hours of rented GPU time without anyone watching, so
the mistakes worth guarding against are the silent ones: a run id it asks for
that the matrix does not build, a configuration key it sets that no longer
exists (the conversion from KV percentage to gigabytes then answers question 3
with the wrong number), or an engine variant whose context window is too small
for the run it is supposed to measure.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stresstest.cli import _largest_context
from stresstest.matrix import BUILDERS
from stresstest.util import load_jsonc
from stresstest.vllm_metrics import METRIC_ALIASES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "pod.sh")
CONFIG = os.path.join(ROOT, "config", "default.json")


def script_text() -> str:
    with open(SCRIPT, encoding="utf-8") as handle:
        return handle.read()


class TestPodScript(unittest.TestCase):
    def test_script_is_executable_and_parses(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK), "scripts/pod.sh is not executable")
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("no bash available")
        result = subprocess.run([bash, "-n", SCRIPT], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_groups_it_runs_exist_in_the_matrix(self):
        """A typo in a group name would silently skip a whole block of runs."""
        groups = set(re.findall(r"^\s*run_group (\w+) ", script_text(), re.M))
        self.assertTrue(groups, "no run_group calls found")
        self.assertLessEqual(groups, set(BUILDERS))
        # Everything in phase 1 is either driven as a group or, for the engine
        # variants, by the loop that restarts the server between runs.
        self.assertEqual(set(BUILDERS) - groups - {"engine", "lesson"}, set())

    def test_engine_variants_map_onto_real_run_ids(self):
        """The wrapper asks for `engine_<name>`; the matrix has to build it."""
        config = load_jsonc(CONFIG)
        built = {spec.run_id for spec in BUILDERS["engine"](config)}
        names = [variant["name"]
                 for variant in config["matrix"]["engine_variation"]["variants"]]
        self.assertTrue(names)
        for name in names:
            self.assertIn(f"engine_{name}", built)

    def test_every_engine_variant_has_room_for_its_own_run(self):
        """--max-model-len below the run's context makes vLLM reject every
        request; the wrapper skips such a variant, so catch it here instead."""
        config = load_jsonc(CONFIG)
        block = config["matrix"]["engine_variation"]
        needed = int(block.get("context_tokens", 32000)) + 1024
        for variant in block["variants"]:
            window = re.search(r"--max-model-len (\d+)", variant.get("flags", ""))
            self.assertIsNotNone(window, f"variant {variant['name']} sets no window")
            self.assertGreaterEqual(
                int(window.group(1)), needed,
                f"variant {variant['name']} measures a {needed}-token run in a "
                f"{window.group(1)}-token window")

    def test_baseline_window_covers_the_heaviest_run(self):
        """The wrapper's default --max-model-len has to fit the whole matrix."""
        default = re.search(r'MAX_MODEL_LEN="\$\{MAX_MODEL_LEN:-(\d+)\}"', script_text())
        self.assertIsNotNone(default)
        self.assertGreaterEqual(int(default.group(1)), _largest_context(load_jsonc(CONFIG)))

    def test_configuration_keys_it_overrides_still_exist(self):
        """These keep the harness aligned with the server it measures. A rename
        in the config would otherwise leave the wrapper writing into nothing."""
        config = load_jsonc(CONFIG)
        keys = set(re.findall(r'--set "([a-z_]+\.[a-z_]+)=', script_text()))
        self.assertIn("hardware.vram_gb", keys)
        self.assertIn("endpoint.model", keys)
        for key in keys:
            section, _, leaf = key.partition(".")
            self.assertIn(section, config, f"unknown config section {section}")
            self.assertIn(leaf, config[section], f"unknown config key {key}")

    def test_metrics_gate_checks_the_series_the_harness_reads(self):
        """Without these three the main question cannot be answered, which is
        why the wrapper refuses to start rather than measure half of it."""
        text = script_text()
        for family in ("num_preemptions", "prefix_cache", "kv_cache_usage_perc"):
            self.assertIn(family, text, f"metrics gate does not look for {family}")
        known = {name for names in METRIC_ALIASES.values() for name in names}
        for family in ("vllm:num_preemptions_total", "vllm:prefix_cache_hits_total",
                       "vllm:kv_cache_usage_perc"):
            self.assertIn(family, known)

    def test_help_lists_every_command_it_accepts(self):
        text = script_text()
        commands = set(re.findall(r"^    (\w[\w-]*)\)\s", text, re.M))
        usage = text.split("usage() {", 1)[1].split("USAGE\n}", 1)[0]
        for command in commands - {"", "all"}:
            self.assertIn(command, usage, f"{command} is not documented in --help")


if __name__ == "__main__":
    unittest.main()
