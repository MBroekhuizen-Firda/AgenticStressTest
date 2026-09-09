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
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
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


HAVE_PROC = os.path.isdir("/proc/self")


def process_is_alive(pid: int) -> bool:
    """True while the process can still hold a file descriptor open, which is
    what holding the run lock comes down to. A zombie cannot: it has been
    through exit(), so its descriptors are closed and its flock released."""
    if HAVE_PROC:
        try:
            with open(f"/proc/{pid}/stat", "rb") as handle:
                # The command name is in brackets and may contain spaces; the
                # state letter is the first field after the closing one.
                return handle.read().rpartition(b")")[2].split()[0] != b"Z"
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True



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
    refuse the second one instead of producing a quietly worthless measurement.

    What holds the lock is the open file descriptor, not the shell: every child
    that inherits fd 9 holds it too. That is the whole design -- a run driving
    the GPU keeps blocking a second one even if its wrapper is gone -- so the
    tests below have to say which of the two they mean, a dead run or a dead
    wrapper, and kill accordingly."""

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

    def start_holder(self):
        """A run holding the lock, with a child of its own holding it too, in a
        session of its own. Returns the shell and the child's pid.

        The child stands in for what the real wrapper runs in the foreground --
        `python3 -m stresstest run` -- which inherits fd 9. It is spawned and
        announced explicitly rather than left to whether bash decides to fork
        for the last command of -c: that difference is what made these tests
        flaky, because whether anything survived the kill was a coin toss."""
        holder = subprocess.Popen(
            ["bash", "-c", self.harness('claim_run\nsleep 30 &\necho "held $!"\nwait\n')],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
        self.addCleanup(self.kill_run, holder)
        announced = holder.stdout.readline().split()
        self.assertEqual(announced[:1], ["held"],
                         "the first run never took the lock")
        return holder, int(announced[1])

    def kill_run(self, holder: subprocess.Popen) -> None:
        """Stop the run the way stopping a pod does: the whole session, not
        just the shell that happens to be its root."""
        try:
            os.killpg(holder.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        holder.wait(timeout=30)
        for pipe in (holder.stdout, holder.stderr):
            pipe.close()

    def wait_until_gone(self, pid: int, timeout: float = 30.0) -> None:
        """SIGKILL stops a process at once, but the kernel closes its file
        descriptors -- and with them releases the flock -- a moment later.
        Asserting on the lock without waiting for that would only trade one
        race for a smaller one."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not process_is_alive(pid):
                return
            time.sleep(0.01)
        self.fail(f"pid {pid} outlived the kill")

    def test_a_second_run_is_refused_while_the_first_holds_the_lock(self):
        holder, _ = self.start_holder()

        second = subprocess.run(
            ["bash", "-c", self.harness('claim_run\necho took-it-anyway\n')],
            capture_output=True, text=True, timeout=60)
        self.assertNotEqual(second.returncode, 0,
                            "a second run started next to a live one: " + second.stdout)
        self.assertIn("er draait al een run", second.stderr)
        # The shell wrote its own $$ into the lock file, and it is still alive,
        # so this pid is the one to point the operator at.
        self.assertIn(str(holder.pid), second.stderr, "the message must name the pid")

    def test_the_lock_dies_with_the_run_that_held_it(self):
        """A run that is gone -- a stopped pod, a killed session -- must not
        leave the next morning's start blocked by a lock nobody holds. Gone
        means the whole session: the wrapper and everything it forked, which is
        what stopping a pod does and what killpg does here."""
        holder, child = self.start_holder()
        self.kill_run(holder)
        self.wait_until_gone(child)

        after = subprocess.run(
            ["bash", "-c", self.harness('claim_run\necho claimed\n')],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(after.returncode, 0, after.stderr)
        self.assertIn("claimed", after.stdout)

    def test_a_run_that_outlives_its_wrapper_still_holds_the_lock(self):
        """The other half of the same rule, and the reason the test above kills
        a whole session. Kill the wrapper alone and the run itself carries on:
        `python3 -m stresstest run` is a foreground child, it inherits fd 9 on
        purpose, and it is still driving the GPU. A second run would be exactly
        as harmful as before, so it stays refused -- an inherited lock here is
        the design, not the leak that the background spawns close fd 9 for."""
        holder, child = self.start_holder()
        holder.kill()
        holder.wait(timeout=30)
        self.assertTrue(process_is_alive(child),
                        "the child died with its shell; this test proves nothing")

        second = subprocess.run(
            ["bash", "-c", self.harness('claim_run\necho took-it-anyway\n')],
            capture_output=True, text=True, timeout=60)
        self.assertNotEqual(
            second.returncode, 0,
            "a second run started next to a live one: " + second.stdout)
        self.assertIn("er draait al een run", second.stderr)

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


class TestHubRateLimit(unittest.TestCase):
    """vLLM asks huggingface.co for the repo's file list on every start, also
    when all 31 GB are already on disk. The Hub throttles anonymous calls per
    IP, and on a rented pod that IP is shared with the neighbours -- so a 429
    ends a start-up that needs nothing from the network at all."""

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _snapshot(self, gigabytes: int) -> str:
        """An HF cache layout with a weights file of the given apparent size.

        Sparse, so a 26 GB download costs no disk here: `du -sb` reports the
        apparent size, which is what the script measures.
        """
        model = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
        snapshot = os.path.join(self.tmp, "hf", "hub",
                                "models--" + model.replace("/", "--"),
                                "snapshots", "deadbeef")
        os.makedirs(snapshot)
        for name in ("config.json", "tokenizer.json"):
            with open(os.path.join(snapshot, name), "w", encoding="utf-8") as handle:
                handle.write("{}\n")
        with open(os.path.join(snapshot, "model.safetensors"), "wb") as handle:
            handle.truncate(gigabytes * 1024 ** 3)
        return snapshot

    def _ask(self, helper: str) -> str:
        """Run one of the model helpers against that fake cache."""
        text = script_text()
        bodies = []
        for name in ("model_dir", "model_size_gb", "model_is_complete",
                     "model_snapshot_dir"):
            found = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.M | re.S)
            self.assertIsNotNone(found, f"{name} is gone")
            bodies.append(found.group(0))
        script = ("set -Eeuo pipefail\n"
                  'MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"\n'
                  f'HF_HOME="{self.tmp}/hf"\n'
                  + "\n".join(bodies) + "\n" + helper)
        done = subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.strip()

    def test_a_complete_download_is_offered_as_the_local_fallback(self):
        snapshot = self._snapshot(26)
        self.assertEqual(self._ask("model_snapshot_dir"), snapshot)

    def test_a_download_that_only_brought_metadata_is_not(self):
        """Serving half a model is worse than saying the Hub is throttling:
        vLLM would fail later, on something that reads like a broken repo."""
        self._snapshot(2)
        self.assertEqual(self._ask("model_snapshot_dir"), "")

    def test_no_download_at_all_is_not_an_error(self):
        """`set -e` plus an empty command substitution ends the whole run, and
        this helper is called on a path where nothing has been fetched yet."""
        self.assertEqual(self._ask("model_snapshot_dir"), "")

    def test_the_retry_recognises_a_throttled_hub(self):
        text = script_text()
        pattern = re.search(r"^HF_RATE_LIMIT_PATTERNS='([^']+)'", text, re.M)
        self.assertIsNotNone(pattern, "the rate-limit pattern is gone")
        self.assertRegex("429 Too Many Requests for url: https://huggingface.co/api",
                         f"(?i){pattern.group(1)}")

    def test_the_retry_does_not_fire_on_every_hub_error(self):
        """A mistyped MODEL gives a Hub error too. Waiting that one out burns a
        quarter of an hour of GPU rent on a typo."""
        text = script_text()
        pattern = re.search(r"^HF_RATE_LIMIT_PATTERNS='([^']+)'", text, re.M)
        self.assertNotRegex("Error retrieving file list: RepositoryNotFoundError",
                            f"(?i){pattern.group(1)}")

    def test_the_flag_survives_the_next_server_start(self):
        """Every engine variant restarts the server. Learning this once per
        start would cost a failed start-up each time."""
        text = script_text()
        self.assertNotIn("local HF_OFFLINE", text,
                         "the flag must outlive a single start_server call")
        self.assertIn("HF_OFFLINE=0\n", text,
                      "the flag needs a default outside start_server")

    def _drive_start_server(self, model_on_disk: bool) -> subprocess.CompletedProcess:
        """Run the real start_server against a vLLM that only ever gets a 429.

        Stubbed: stop_server (nothing to stop) and wait_ready (the fake server
        announces itself in the log instead of on a port). Everything the test
        is about -- the fallback, the retry, the diagnosis -- is the script's.
        """
        home = os.path.join(self.tmp, "hf")
        if model_on_disk:
            self._snapshot(26)
        bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        vllm = os.path.join(bin_dir, "vllm")
        with open(vllm, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent("""\
                #!/usr/bin/env bash
                # A repo id goes to the Hub and is throttled; a path does not.
                case "$2" in
                  # exec, so the pid the script records is the one to kill.
                  /*) echo "Application startup complete."; exec sleep 5 ;;
                  *)  echo "ERROR [repo_utils.py:117] 429 Too Many Requests for url:"
                      echo "  https://huggingface.co/api/models/$2/tree/main"
                      exit 1 ;;
                esac
                """))
        os.chmod(vllm, os.stat(vllm).st_mode | stat.S_IEXEC)

        text = script_text()
        bodies = []
        for name in ("model_dir", "model_size_gb", "model_is_complete",
                     "model_snapshot_dir", "show_server_error",
                     "die_server_start", "start_server"):
            found = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.M | re.S)
            self.assertIsNotNone(found, f"{name} is gone")
            bodies.append(found.group(0))
        consts = [line for line in text.splitlines()
                  if line.startswith(("SERVER_ERROR_PATTERNS=", "HF_RATE_LIMIT_PATTERNS="))]
        state = os.path.join(self.tmp, "state")
        os.makedirs(state, exist_ok=True)
        script = "\n".join([
            "set -Eeuo pipefail",
            'MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"',
            "SERVED_NAME=qwen3-coder; HOST_BIND=127.0.0.1; PORT=8000",
            "GPU_UTIL=0.90; TENSOR_PARALLEL=1; MOCK=0",
            "KV_CACHE_DTYPE=fp8; MAX_NUM_SEQS=32; MAX_MODEL_LEN=131072",
            'ATTENTION_BACKEND=""; NO_FLASHINFER_SAMPLER=0',
            "HF_OFFLINE=0; SERVER_START_ATTEMPTS=2",
            "HF_RATE_LIMIT_WAIT_S=1; HF_RATE_LIMIT_MAX_WAIT_S=1",
            "SERVER_START_TIMEOUT_S=900",
            f'HF_HOME="{home}"',
            f'STATE_DIR="{state}"',
            'SERVER_LOG="$STATE_DIR/vllm.log"',
            'SERVER_PID_FILE="$STATE_DIR/vllm.pid"',
            *consts,
            'say()  { echo "[say] $*" >&2; }',
            'warn() { echo "[warn] $*" >&2; }',
            'die()  { echo "[die] $*" >&2; exit 1; }',
            "stop_server() { :; }",
            "wait_ready() {",
            "  sleep 0.3",
            '  grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null && return 0',
            "  return 2",
            "}",
            *bodies,
            "start_server",
            'echo "[ok] HF_OFFLINE=$HF_OFFLINE"',
        ])
        env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"])
        done = subprocess.run(["bash", "-c", script], env=env,
                              capture_output=True, text=True, timeout=180)
        # A start that succeeded left the fake server running, exactly as the
        # real one does. Nothing here waits for it, so take it down.
        self.addCleanup(self._kill, os.path.join(state, "vllm.pid"))
        return done

    @staticmethod
    def _kill(pid_file: str) -> None:
        try:
            with open(pid_file, encoding="utf-8") as handle:
                pid = int(handle.read().strip())
        except (OSError, ValueError):
            return
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    def test_a_throttled_hub_falls_back_to_the_weights_on_disk(self):
        done = self._drive_start_server(model_on_disk=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[ok] HF_OFFLINE=1", done.stdout)
        self.assertIn("snapshots/deadbeef --served-model-name", done.stderr,
                      "the second attempt still passed the repo id")

    def test_a_throttled_hub_is_not_reported_as_a_memory_problem(self):
        """Two failures in one: reporting success on a server that never came
        up, and blaming the KV cache for the neighbours' traffic. The first
        would send the harness at a dead port; the second would send the
        operator to lower --max-model-len, which changes nothing."""
        done = self._drive_start_server(model_on_disk=False)
        self.assertNotEqual(done.returncode, 0,
                            "start_server reported success without a server")
        self.assertIn("429", done.stderr)
        self.assertNotIn("te weinig geheugen voor de KV-cache", done.stderr)


class TestPushingResults(unittest.TestCase):
    """The results only become safe once they leave the rented machine, and
    the pod is only allowed to stop after that has happened."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _repo(self, with_remote: bool) -> str:
        """A throwaway clone with results/ ignored, like the real one."""
        origin = os.path.join(self.tmp, "origin.git")
        subprocess.run(["git", "init", "--bare", "-q", origin], check=True)
        repo = os.path.join(self.tmp, "repo")
        subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
        with open(os.path.join(repo, ".gitignore"), "w", encoding="utf-8") as handle:
            handle.write("results/\n")
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "-C", repo, "add", ".gitignore"], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-qm", "init"], check=True, env=env)
        if with_remote:
            subprocess.run(["git", "-C", repo, "remote", "add", "origin", origin],
                           check=True)
        os.makedirs(os.path.join(repo, "results", "20260909_matrix"))
        with open(os.path.join(repo, "results", "20260909_matrix", "RESULTATEN.md"),
                  "w", encoding="utf-8") as handle:
            handle.write("# meting\n")
        return repo, origin

    def _harness(self, repo: str, tail: str) -> str:
        text = script_text()
        bodies = []
        for name in ("setup_git_credentials", "push_results"):
            found = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.M | re.S)
            self.assertIsNotNone(found, f"{name} is gone")
            bodies.append(found.group(0))
        return ("set -Eeuo pipefail\n"
                f'REPO_DIR="{repo}"\n'
                f'STATE_DIR="{self.tmp}/state"\n'
                'GPU_NAME="NVIDIA RTX PRO 6000 Blackwell (600W)"\n'
                'PUSH_RETRIES=1\n'
                'RESULTS_BRANCH=""\n'
                'say()   { echo "$*"; }\n'
                'warn()  { echo "LET OP: $*" >&2; }\n'
                'head_() { echo "$*"; }\n'
                + "\n".join(bodies) + "\n" + tail)

    def test_results_are_pushed_even_though_gitignore_excludes_them(self):
        repo, origin = self._repo(with_remote=True)
        done = subprocess.run(
            ["bash", "-c", self._harness(repo, 'push_results results/20260909_matrix\n')],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        listing = subprocess.run(
            ["git", "--git-dir", origin, "log", "--all", "--name-only", "--format="],
            capture_output=True, text=True, check=True).stdout
        self.assertIn("results/20260909_matrix/RESULTATEN.md", listing,
                      "results/ is in .gitignore, so it needs `git add -f`")

    def test_a_missing_remote_fails_instead_of_reporting_success(self):
        """A push that silently did nothing would let the pod stop on top of
        results that exist nowhere else."""
        repo, _ = self._repo(with_remote=False)
        done = subprocess.run(
            ["bash", "-c", self._harness(repo, 'push_results results/20260909_matrix\n')],
            capture_output=True, text=True, timeout=120)
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertIn("origin", done.stderr)

    def test_a_token_from_the_environment_is_enough_to_push(self):
        """The RunPod-secret route: the operator stores the token as a secret,
        the template puts it in GITHUB_TOKEN, and nobody types it anywhere. The
        remote itself carries no credentials, and neither does anything on
        disk."""
        repo, _ = self._repo(with_remote=True)
        subprocess.run(["git", "-C", repo, "remote", "set-url", "origin",
                        "https://github.com/example/repo.git"], check=True)
        done = subprocess.run(
            ["bash", "-c", self._harness(
                repo,
                'setup_git_credentials "$(git -C "$REPO_DIR" remote get-url origin)"\n'
                '"${GIT_ASKPASS}" "Username for https://github.com"\n'
                '"${GIT_ASKPASS}" "Password for https://github.com"\n')],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "GITHUB_TOKEN": "ghp_niet_echt"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("x-access-token", done.stdout)
        self.assertIn("ghp_niet_echt", done.stdout,
                      "the helper must hand git the token from the environment")

        # And it must not have been written down anywhere.
        config = subprocess.run(["git", "-C", repo, "config", "--list"],
                                capture_output=True, text=True, check=True).stdout
        self.assertNotIn("ghp_niet_echt", config, "the token ended up in .git/config")
        remote = subprocess.run(["git", "-C", repo, "remote", "get-url", "origin"],
                                capture_output=True, text=True, check=True).stdout
        self.assertNotIn("ghp_niet_echt", remote, "the token ended up in the remote URL")
        with open(os.path.join(self.tmp, "state", "git-askpass.sh"), encoding="utf-8") as h:
            self.assertNotIn("ghp_niet_echt", h.read(),
                             "the token was baked into the helper script")

    def test_a_remote_that_already_has_credentials_is_left_alone(self):
        """Someone who put the token in the URL themselves should not have
        their setup quietly replaced."""
        repo, _ = self._repo(with_remote=True)
        subprocess.run(["git", "-C", repo, "remote", "set-url", "origin",
                        "https://x-access-token:abc@example.invalid/x/y.git"], check=True)
        done = subprocess.run(
            ["bash", "-c", self._harness(
                repo,
                'setup_git_credentials "$(git -C "$REPO_DIR" remote get-url origin)"\n'
                'echo "askpass=${GIT_ASKPASS:-geen}"\n')],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "GITHUB_TOKEN": "ghp_niet_echt"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("askpass=geen", done.stdout)

    def test_push_access_is_checked_before_the_measurement_not_after(self):
        """Six hours of GPU rental followed by 'kan niet pushen' is the failure
        this check exists to prevent, and the hint has to name the RunPod
        template variable because that is the step people miss."""
        text = script_text()
        body = re.search(r'^cmd_all\(\) \{.*?^\}', text, re.M | re.S)
        self.assertIsNotNone(body, "cmd_all is gone")
        tail = body.group(0)
        self.assertIn("check_push_access", tail)
        # Before any measurement group runs.
        self.assertLess(tail.index("check_push_access"), tail.index("run_group rampup"),
                        "the push check must come before the first run")
        hint = re.search(r'^_push_access_hint\(\) \{.*?^\}', text, re.M | re.S)
        self.assertIsNotNone(hint, "_push_access_hint is gone")
        self.assertIn("GITHUB_TOKEN = {{ RUNPOD_SECRET_", hint.group(0),
                      "the hint must show the template reference verbatim")
        self.assertIn("--no-push", hint.group(0))

    def test_the_pod_only_stops_after_a_successful_push(self):
        text = script_text()
        body = re.search(r'^cmd_all\(\) \{.*?^\}', text, re.M | re.S)
        self.assertIsNotNone(body, "cmd_all is gone")
        tail = body.group(0)
        self.assertIn('push_results "$dir" "$lesson_dir" && pushed=1', tail,
                      "the shutdown must hang on the push actually succeeding")
        self.assertIn('if [ "$pushed" = 1 ] || [ "$SHUTDOWN_WHEN_DONE" = 1 ]', tail)
        # A failed push must not disarm the deadman: the pod stays up so the
        # results can be fetched, but not forever.
        failure_branch = tail.split('elif [ "$PUSH_RESULTS" = 1 ]; then')[1]
        self.assertNotIn("disarm_deadman", failure_branch.split("else")[0])


if __name__ == "__main__":
    unittest.main()
