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
import stat
import subprocess
import sys
import tempfile
import textwrap
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


class TestCardIdentification(unittest.TestCase):
    """The RTX PRO 6000 comes in variants that carry the same 96 GB but not the
    same power budget. Measuring on a Max-Q and reporting it as the card in the
    funding request would be a quiet, expensive mistake, so the wrapper reads
    the power limit and records it."""

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        stub = os.path.join(self.tmp, "nvidia-smi")
        with open(stub, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent("""\
                #!/usr/bin/env bash
                q=""
                for a in "$@"; do case "$a" in --query-gpu=*) q="${a#--query-gpu=}";; esac; done
                case "$q" in
                  name,driver_version,memory.total) echo "$FAKE_GPU_NAME, 580.65.06, 97887 MiB" ;;
                  driver_version) echo "580.65.06" ;;
                  name,memory.total) echo "$FAKE_GPU_NAME, 97887" ;;
                  name) echo "$FAKE_GPU_NAME" ;;
                  power.default_limit) echo "$FAKE_WATTS" ;;
                  *) exit 1 ;;
                esac
                """))
        os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)

    def plan_stderr(self, name: str, watts: str) -> str:
        env = dict(os.environ,
                   PATH=self.tmp + os.pathsep + os.environ["PATH"],
                   FAKE_GPU_NAME=name, FAKE_WATTS=watts)
        result = subprocess.run(["bash", SCRIPT, "plan"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        return result.stderr

    def test_max_q_is_called_out(self):
        stderr = self.plan_stderr("NVIDIA RTX PRO 6000 Blackwell Max-Q", "300.00")
        self.assertIn("Max-Q", stderr)

    def test_full_power_card_passes_without_a_warning(self):
        stderr = self.plan_stderr("NVIDIA RTX PRO 6000 Blackwell Server Edition", "600.00")
        self.assertNotIn("Max-Q", stderr)

    def test_a_driver_without_a_power_limit_is_not_fatal(self):
        stderr = self.plan_stderr("NVIDIA RTX PRO 6000", "[N/A]")
        self.assertNotIn("Max-Q", stderr)
        self.assertNotIn("FOUT", stderr)


class TestFirstCleanStart(unittest.TestCase):
    """The very first run has no results/ directory yet. `ls` fails there, and
    under `set -o pipefail` that failure escapes the command substitution and
    aborts the whole run -- after the model has already been downloaded. Every
    earlier test passed RESULTS_DIR explicitly and so never took this branch."""

    def test_helper_returns_empty_instead_of_failing(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")
        body = re.search(r"^latest_matrix_dir\(\) \{.*?^\}", script_text(), re.M | re.S)
        self.assertIsNotNone(body, "latest_matrix_dir is gone")
        empty = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, empty, True)
        program = (f"set -Eeuo pipefail\ncd {empty}\n{body.group(0)}\n"
                   'directory="$(latest_matrix_dir)"\n'
                   'echo "reached-the-end[$directory]"\n')
        result = subprocess.run(["bash", "-c", program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reached-the-end[]", result.stdout)

    def test_every_results_listing_goes_through_the_helper(self):
        lines = [line for line in script_text().splitlines() if "ls -dt results" in line]
        self.assertEqual(len(lines), 1,
                         "a results listing outside latest_matrix_dir will abort "
                         f"on a clean tree: {lines}")
        self.assertIn("|| true", lines[0])


class TestOnlyOneRunAtATime(unittest.TestCase):
    """Two runs side by side share one GPU, one port and one results directory.
    Nothing in the numbers afterwards says that happened, so the wrapper has to
    refuse the second one instead of producing a quietly worthless measurement."""

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def harness(self, tail: str) -> str:
        """The lock functions, lifted from the script, over a throwaway state
        directory. Running the real script would mean a real preflight."""
        text = script_text()
        bodies = []
        for name in ("running_run_pid", "die_second_run", "claim_run"):
            found = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.M | re.S)
            self.assertIsNotNone(found, f"{name} is gone")
            bodies.append(found.group(0))
        return ("set -Eeuo pipefail\n"
                f'STATE_DIR="{self.tmp}"\n'
                'RUN_LOCK_FILE="$STATE_DIR/run.lock"\n'
                'say()  { echo "$*"; }\n'
                'die()  { echo "FOUT: $*" >&2; exit 1; }\n'
                + "\n".join(bodies) + "\n" + tail)

    def test_a_second_run_is_refused_while_the_first_holds_the_lock(self):
        holder = subprocess.Popen(
            ["bash", "-c", self.harness('claim_run\necho held\nsleep 30\n')],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(holder.kill)
        self.assertEqual(holder.stdout.readline().strip(), "held",
                         "the first run never took the lock")

        second = subprocess.run(
            ["bash", "-c", self.harness('claim_run\necho took-it-anyway\n')],
            capture_output=True, text=True, timeout=60)
        self.assertNotEqual(second.returncode, 0,
                            "a second run started next to a live one: " + second.stdout)
        self.assertIn("er draait al een run", second.stderr)
        self.assertIn(str(holder.pid), second.stderr, "the message must name the pid")

    def test_the_lock_dies_with_the_run_that_held_it(self):
        """A killed run, or a pod stopped mid-run, must not leave the next
        morning's start blocked by a lock nobody holds."""
        holder = subprocess.Popen(
            ["bash", "-c", self.harness('claim_run\necho held\nsleep 30\n')],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(holder.kill)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        holder.kill()
        holder.wait(timeout=30)

        after = subprocess.run(
            ["bash", "-c", self.harness('claim_run\necho claimed\n')],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(after.returncode, 0, after.stderr)
        self.assertIn("claimed", after.stdout)

    def test_background_children_do_not_inherit_the_lock(self):
        """The deadman outlives the run on purpose. If it inherited the lock
        file descriptor it would hold the lock until it fires, and the next
        start would be refused for no reason."""
        spawns = [line for line in script_text().splitlines()
                  if "nohup" in line and not line.lstrip().startswith("#")
                  and "--detach " not in line]
        self.assertTrue(spawns, "no background spawns found -- has the script moved on?")
        for line in spawns:
            self.assertIn("9>&-", line,
                          f"this child inherits the run lock: {line.strip()}")

    def test_the_commands_that_drive_a_run_claim_the_lock(self):
        text = script_text()
        claiming = re.search(r"^\s*(\S+)\) claim_run ;;", text, re.M)
        self.assertIsNotNone(claiming, "no command claims the lock any more")
        for command in ("all", "serve", "group", "lesson"):
            self.assertIn(command, claiming.group(1),
                          f"{command} starts a server without taking the lock")


class TestFlashInferFallback(unittest.TestCase):
    """vLLM 0.28 reaches for FlashInfer for top-k/top-p sampling whatever the
    attention backend is. On this card its JIT compiler cannot build, and with
    the package removed vLLM's own import of it fails instead -- both end the
    run. Sampling without it changes nothing here: the harness runs at
    temperature 0."""

    def test_a_missing_flashinfer_module_triggers_the_retry(self):
        text = script_text()
        retry = re.search(r"NO_FLASHINFER_SAMPLER\" != 1.*?grep -qEi (\"[^\"]+\")",
                          text, re.S)
        self.assertIsNotNone(retry, "the FlashInfer retry is gone")
        pattern = retry.group(1)
        self.assertIn("No module named", pattern,
                      "an absent flashinfer is the other way this fails")

    def test_the_flag_survives_the_next_server_start(self):
        """Every engine variant restarts the server. Learning this once per
        start would cost a failed start-up each time."""
        text = script_text()
        self.assertNotIn("local no_flashinfer_sampler", text,
                         "the flag must outlive a single start_server call")
        self.assertIn("NO_FLASHINFER_SAMPLER=0\n", text,
                      "the flag needs a default outside start_server")


if __name__ == "__main__":
    unittest.main()
