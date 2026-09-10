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


def function_body(name: str) -> str:
    """One shell function out of the script, so a test can read what it does
    without running seven hours of GPU time to find out."""
    found = re.search(rf"^{name}\(\) \{{.*?^\}}", script_text(), re.M | re.S)
    assert found is not None, f"{name} is gone"
    return found.group(0)


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


class TestDescribingAResumedDirectory(unittest.TestCase):
    """The same trap as TestFirstCleanStart, one directory deeper. A resumed
    directory whose first server never came up exists but has no runs/ in it
    yet; `find` exits 1 on that, pipefail hands the failure on, and the ERR
    trap ends the run with "afgebroken op regel N" -- over a line that was
    only counting subdirectories for a status message."""

    def _describe(self, directory: str) -> subprocess.CompletedProcess:
        program = "\n".join([
            "set -Eeuo pipefail",
            'die() { trap - ERR; printf "FOUT: %s\\n" "$*" >&2; exit 1; }',
            'on_error() { die "afgebroken op regel $1."; }',
            "trap 'on_error $LINENO' ERR",
            'say() { printf "%s\\n" "$*" >&2; }',
            function_body("plural_runs"),
            function_body("describe_results_dir"),
            f'say "hervat in bestaande map {directory} ($(describe_results_dir {directory}))"',
            "echo reached-the-end",
        ])
        return subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_directory_without_runs_yet_does_not_abort_the_run(self):
        directory = os.path.join(self.tmp, "20260910-081024_matrix")
        os.makedirs(directory)
        done = self._describe(directory)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("reached-the-end", done.stdout)
        self.assertNotIn("FOUT", done.stderr)
        self.assertIn("0 runs aanwezig", done.stderr)

    def test_a_directory_with_runs_is_still_counted(self):
        directory = os.path.join(self.tmp, "20260910-081024_matrix")
        os.makedirs(os.path.join(directory, "runs", "a"))
        os.makedirs(os.path.join(directory, "runs", "b"))
        done = self._describe(directory)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("2 runs aanwezig", done.stderr)


class TestTensorParallelIsCheckedBeforeTheRun(unittest.TestCase):
    """--tensor-parallel-size is a promise about the machine: one worker
    process per card, all of them talking over shared memory. Both ways that
    promise is broken are answerable in a second here, and cost a quarter of
    an hour of rent and an unreadable traceback if they are left to vLLM."""

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")

    def _check(self, tensor_parallel: int, gpu_count: int, force: int = 0,
               shm_mb: str = "8192") -> subprocess.CompletedProcess:
        program = "\n".join([
            "set -Eeuo pipefail",
            'say() { echo "[say] $*" >&2; }',
            'warn() { echo "[warn] $*" >&2; }',
            'die() { echo "[die] $*" >&2; exit 1; }',
            f'shm_size_mb() {{ printf "%s" "{shm_mb}"; }}',
            f"TENSOR_PARALLEL={tensor_parallel}; GPU_COUNT={gpu_count}; FORCE={force}",
            function_body("check_tensor_parallel"),
            "check_tensor_parallel",
            "echo reached-the-end",
        ])
        return subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    def test_more_workers_than_cards_is_refused_before_anything_is_loaded(self):
        done = self._check(tensor_parallel=2, gpu_count=1)
        self.assertEqual(done.returncode, 1)
        self.assertIn("[die]", done.stderr)
        self.assertIn("TENSOR_PARALLEL=2", done.stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES", done.stderr)

    def test_force_lets_the_operator_through_but_says_so(self):
        done = self._check(tensor_parallel=2, gpu_count=1, force=1)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[warn]", done.stderr)
        self.assertIn("reached-the-end", done.stdout)

    def test_a_container_without_shared_memory_is_warned_about(self):
        """64 MB is what a container without --shm-size gets, and vLLM's
        workers cannot build their channels in it. Nothing on the machine
        says so; the failure arrives as "WorkerProc initialization failed"."""
        done = self._check(tensor_parallel=2, gpu_count=2, shm_mb="64")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("shm-size", done.stderr)
        self.assertIn("reached-the-end", done.stdout)

    def test_one_card_is_never_bothered_with_any_of_this(self):
        done = self._check(tensor_parallel=1, gpu_count=1, shm_mb="64")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("[warn]", done.stderr)

    def test_the_check_runs_in_preflight(self):
        self.assertIn("check_tensor_parallel", function_body("preflight"))

    def test_the_pool_never_counts_cards_that_are_not_there(self):
        """VRAM_GB is what the harness converts KV percentages to gigabytes
        with. Multiplying one card by a TENSOR_PARALLEL of two reports a
        192 GB machine and puts every gigabyte in the report out by two."""
        body = function_body("detect_gpu")
        program = "\n".join([
            "set -Eeuo pipefail",
            'warn() { :; }',
            "TENSOR_PARALLEL=2",
            'nvidia_smi_one_card() { :; }',
            body.replace("nvidia-smi", "fake_smi"),
            "command() { return 0; }",
            'fake_smi() {',
            '  case "$*" in',
            '    *name,memory.total*) echo "NVIDIA RTX PRO 6000, 97887" ;;',
            '    *power.default_limit*) echo "600.00" ;;',
            '    *name*) echo "NVIDIA RTX PRO 6000" ;;',
            '  esac',
            '}',
            "detect_gpu",
            'echo "VRAM_GB=$VRAM_GB GPU_COUNT=$GPU_COUNT"',
        ])
        done = subprocess.run(["bash", "-c", program], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("GPU_COUNT=1", done.stdout)
        pool = float(done.stdout.split("VRAM_GB=")[1].split()[0])
        self.assertAlmostEqual(pool, 95.6, delta=0.5,
                               msg=f"one 96 GB card, not two: {done.stdout}")


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


THROTTLED_VLLM = """\
#!/usr/bin/env bash
# A repo id goes to the Hub and is throttled; a path does not.
case "$2" in
  # exec, so the pid the script records is the one to kill.
  /*) echo "Application startup complete."; exec sleep 5 ;;
  *)  echo "ERROR [repo_utils.py:117] 429 Too Many Requests for url:"
      echo "  https://huggingface.co/api/models/$2/tree/main"
      exit 1 ;;
esac
"""


class StartServerHarness(unittest.TestCase):
    """Drives the script's real start_server against a fake `vllm` on PATH.

    Stubbed: stop_server (nothing to stop) and wait_ready (the fake server
    announces itself in the log instead of on a port). Everything the tests
    are about -- the fallbacks, the retries, the diagnosis -- is the script's.
    """

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

    def _drive_start_server(self, fake_vllm: str, model_on_disk: bool = True,
                            attempts: int = 4, backend: str = "",
                            tensor_parallel: int = 1) -> subprocess.CompletedProcess:
        """Run start_server with `fake_vllm` (a bash script) as vllm.
        `backend` is what an operator would export as ATTENTION_BACKEND."""
        home = os.path.join(self.tmp, "hf")
        if model_on_disk:
            self._snapshot(26)
        bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        vllm = os.path.join(bin_dir, "vllm")
        with open(vllm, "w", encoding="utf-8") as handle:
            handle.write(fake_vllm)
        os.chmod(vllm, os.stat(vllm).st_mode | stat.S_IEXEC)

        text = script_text()
        bodies = []
        for name in ("model_dir", "model_size_gb", "model_is_complete",
                     "model_snapshot_dir", "shm_size_mb", "driver_cuda_version",
                     "log_suspects",
                     "worker_lines", "show_server_error",
                     "die_server_start", "start_server"):
            found = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.M | re.S)
            self.assertIsNotNone(found, f"{name} is gone")
            bodies.append(found.group(0))
        # Every pattern constant, so a new one cannot be missed here and read
        # as an unbound variable under set -u.
        consts = [line for line in text.splitlines()
                  if re.match(r"^[A-Z_]+(_PATTERNS|_STRIP)=", line)]
        state = os.path.join(self.tmp, "state")
        os.makedirs(state, exist_ok=True)
        script = "\n".join([
            "set -Eeuo pipefail",
            'MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"',
            "SERVED_NAME=qwen3-coder; HOST_BIND=127.0.0.1; PORT=8000",
            f"GPU_UTIL=0.90; TENSOR_PARALLEL={tensor_parallel}; GPU_COUNT={tensor_parallel}; MOCK=0",
            "KV_CACHE_DTYPE=fp8; MAX_NUM_SEQS=32; MAX_MODEL_LEN=131072",
            f'ATTENTION_BACKEND="{backend}"; NO_FLASHINFER_SAMPLER=0',
            "NCCL_P2P_OFF=0",
            f"HF_OFFLINE=0; SERVER_START_ATTEMPTS={attempts}",
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
            'echo "[ok] ATTENTION_BACKEND=$ATTENTION_BACKEND"',
            'echo "[ok] NO_FLASHINFER_SAMPLER=$NO_FLASHINFER_SAMPLER"',
            'echo "[ok] NCCL_P2P_OFF=$NCCL_P2P_OFF"',
            'echo "[ok] current_backend=$(cat "$STATE_DIR/current_backend" 2>/dev/null)"',
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

    @staticmethod
    def starts(done: subprocess.CompletedProcess) -> list:
        """The vllm command lines the script tried, in order."""
        return [line.split("start: ", 1)[1] for line in done.stderr.splitlines()
                if "[say] start: " in line]


# vLLM's start-up log echoes every non-default argument. That line is in every
# log, including one of a server that died on something else entirely.
ARGS_ECHO = ("INFO [utils.py:253] non-default args: {'enable_auto_tool_choice': True,"
             " 'tool_call_parser': 'qwen3_coder', 'kv_cache_dtype': 'fp8'}")

# The two ways FlashInfer's JIT compiler ends a start on this card. Same root
# cause -- "SM 12.x requires CUDA >= 12.9", after which check_cuda_arch()
# reports the misleading sm75 message -- reached through two different
# callers: the attention backend building its prefill module, or the sampler.
ATTENTION_JIT_FAILURE = """\
echo "$ARGS_ECHO"
echo "INFO [flashinfer.py:190] Using FlashInfer backend on V1 engine."
echo "ERROR [core.py:1374]   File \\"/dist-packages/vllm/v1/attention/backends/flashinfer.py\\", line 900, in _plan"
echo "ERROR [core.py:1374]   File \\"/dist-packages/flashinfer/jit/attention/modules.py\\", line 1735, in gen_customize_batch_prefill_module"
echo "ERROR [core.py:1374]   File \\"/dist-packages/flashinfer/jit/core.py\\", line 109, in check_cuda_arch"
echo "ERROR [core.py:1374] RuntimeError: FlashInfer requires GPUs with sm75 or higher"
echo "RuntimeError: Engine core initialization failed. See root cause above."
exit 1
"""
SAMPLER_JIT_FAILURE = """\
echo "$ARGS_ECHO"
echo "INFO [topk_topp_sampler.py:62] Using FlashInfer for top-p and top-k sampling."
echo "ERROR [core.py:1374]   File \\"/dist-packages/vllm/v1/attention/backends/flashinfer.py\\", line 20, in <module>"
echo "ERROR [core.py:1374]   File \\"/dist-packages/flashinfer/sampling.py\\", line 70, in get_sampling_module"
echo "ERROR [core.py:1374]   File \\"/dist-packages/flashinfer/jit/core.py\\", line 109, in check_cuda_arch"
echo "ERROR [core.py:1374] RuntimeError: FlashInfer requires GPUs with sm75 or higher"
exit 1
"""
STARTED = 'echo "Application startup complete."; exec sleep 5'


def shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def fake_vllm(body: str) -> str:
    """A `vllm` that runs `body` with the arguments in $* and the sampler
    variable in $SAMPLER; `$ARGS_ECHO` is vLLM's argument echo."""
    return ("#!/usr/bin/env bash\n"
            f"ARGS_ECHO={shell_quote(ARGS_ECHO)}\n"
            'SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-}"\n'
            + textwrap.dedent(body))


# A vLLM that only starts on TRITON_ATTN, and dies in FlashInfer's attention
# JIT otherwise: the pod as it was on the day the fallback was written.
NEEDS_TRITON = fake_vllm(f"""
    case " $* " in
      *" --attention-backend TRITON_ATTN "*) {STARTED} ;;
    esac
    {ATTENTION_JIT_FAILURE}""")


class TestFlashInferFallback(StartServerHarness):
    """FlashInfer's JIT compiler cannot build for this card with the toolkit
    on the image, and vLLM 0.28 reaches for FlashInfer in two places: the
    top-k/top-p sampler, whatever the attention backend, and the attention
    backend itself once --kv-cache-dtype fp8 rules FLASH_ATTN out. The
    script has to steer around both, without dropping anything else on the
    way -- the tool-call flags in particular -- and has to say what it did."""

    def test_the_attention_backend_is_switched_when_the_log_blames_it(self):
        """The case from the field: the sampler was already off and the
        engine still died building FlashInfer's prefill module. That start-up
        cannot be rescued by the sampler; only another backend helps."""
        done = self._drive_start_server(NEEDS_TRITON)
        self.assertEqual(done.returncode, 0, done.stderr)
        starts = self.starts(done)
        self.assertEqual(len(starts), 2,
                         "the sampler and the backend should be switched in one go:\n"
                         + done.stderr)
        self.assertIn("--attention-backend TRITON_ATTN", starts[-1])
        self.assertIn("[ok] current_backend=TRITON_ATTN", done.stdout,
                      "the backend must reach the results as hardware.attention_backend")
        self.assertIn("[ok] ATTENTION_BACKEND=TRITON_ATTN", done.stdout,
                      "the next engine variant would otherwise burn a start-up on it again")
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER=0", done.stderr)

    def test_the_tool_call_flags_survive_a_flashinfer_failure(self):
        """What actually happened on the pod: the second FlashInfer failure
        fell through to the tool-parser fallback, because vLLM's argument echo
        contains 'tool_call_parser'. The engine died again in the same way,
        and had it started, the harness would have measured without tool
        calling."""
        done = self._drive_start_server(NEEDS_TRITON)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("tool-call-parser wordt niet geaccepteerd", done.stderr)
        for command in self.starts(done):
            self.assertIn("--tool-call-parser qwen3_coder", command)

    def test_a_sampler_only_failure_leaves_the_backend_alone(self):
        """Another backend changes the numbers; the sampler does not. When
        the log only blames the sampler, that is all that should change."""
        done = self._drive_start_server(fake_vllm(f"""
            [ "$SAMPLER" = 0 ] && {{ {STARTED}; }}
            {SAMPLER_JIT_FAILURE}"""))
        self.assertEqual(done.returncode, 0, done.stderr)
        starts = self.starts(done)
        self.assertEqual(len(starts), 2, done.stderr)
        self.assertNotIn("--attention-backend", starts[-1])
        self.assertIn("[ok] current_backend=\n", done.stdout)

    def test_a_backend_the_operator_chose_is_not_substituted(self):
        """ATTENTION_BACKEND=FLASHINFER from the outside is a decision, not a
        default. The sampler may still be switched off; the backend may not."""
        done = self._drive_start_server(NEEDS_TRITON, backend="FLASHINFER")
        self.assertNotEqual(done.returncode, 0)
        for command in self.starts(done):
            self.assertIn("--attention-backend FLASHINFER", command)
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER=0", done.stderr)

    def test_a_card_that_cannot_be_helped_gets_the_toolkit_diagnosis(self):
        """Sampler off, TRITON_ATTN, and still FlashInfer: nothing left to
        try. Then say so -- and do not blame the KV cache, which is where the
        generic message sends people."""
        done = self._drive_start_server(fake_vllm(ATTENTION_JIT_FAILURE))
        self.assertNotEqual(done.returncode, 0, "start_server reported success without a server")
        self.assertLessEqual(len(self.starts(done)), 2,
                             "a hopeless start-up should not be repeated:\n" + done.stderr)
        self.assertIn("12.9", done.stderr)
        self.assertNotIn("te weinig geheugen voor de KV-cache", done.stderr)
        self.assertNotIn("tool-call-parser wordt niet geaccepteerd", done.stderr)

    def test_an_old_vllm_without_the_flag_is_named(self):
        done = self._drive_start_server(fake_vllm(f"""
            case " $* " in
              *" --attention-backend "*)
                echo "vllm serve: error: unrecognized arguments: --attention-backend TRITON_ATTN"
                exit 2 ;;
            esac
            {ATTENTION_JIT_FAILURE}"""))
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("--attention-backend niet", done.stderr)
        self.assertNotIn("tool-call-parser wordt niet geaccepteerd", done.stderr)

    def test_the_tool_parser_fallback_still_fires_on_a_real_refusal(self):
        done = self._drive_start_server(fake_vllm(f"""
            case " $* " in
              *" --tool-call-parser "*)
                echo "$ARGS_ECHO"
                echo "ValueError: invalid tool call parser: qwen3_coder (chose from {{hermes}})"
                exit 1 ;;
            esac
            {STARTED}"""))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("tool-call-parser wordt niet geaccepteerd", done.stderr)
        self.assertNotIn("--tool-call-parser", self.starts(done)[-1])

    def test_a_missing_flashinfer_module_triggers_the_retry(self):
        text = script_text()
        pattern = re.search(r"^FLASHINFER_PATTERNS='([^']+)'", text, re.M)
        self.assertIsNotNone(pattern, "the FlashInfer pattern is gone")
        self.assertRegex("ModuleNotFoundError: No module named 'flashinfer'",
                         f"(?i){pattern.group(1)}")

    def test_the_backend_is_a_server_argument_not_the_dead_variable(self):
        """vLLM 0.28 no longer reads VLLM_ATTENTION_BACKEND: it warns about an
        unknown variable and picks FlashInfer anyway."""
        text = script_text()
        self.assertNotIn('VLLM_ATTENTION_BACKEND="$backend"', text)
        self.assertIn('--attention-backend $backend', text)

    def test_the_flags_survive_the_next_server_start(self):
        """Every engine variant restarts the server. Learning this once per
        start would cost a failed start-up each time."""
        text = script_text()
        self.assertNotIn("local no_flashinfer_sampler", text,
                         "the flag must outlive a single start_server call")
        self.assertIn("NO_FLASHINFER_SAMPLER=0\n", text,
                      "the flag needs a default outside start_server")
        self.assertNotIn("local ATTENTION_BACKEND", text)


class TestHubRateLimit(StartServerHarness):
    """vLLM asks huggingface.co for the repo's file list on every start, also
    when all 31 GB are already on disk. The Hub throttles anonymous calls per
    IP, and on a rented pod that IP is shared with the neighbours -- so a 429
    ends a start-up that needs nothing from the network at all."""

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

    def test_a_throttled_hub_falls_back_to_the_weights_on_disk(self):
        done = self._drive_start_server(THROTTLED_VLLM, model_on_disk=True, attempts=2)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[ok] HF_OFFLINE=1", done.stdout)
        self.assertIn("snapshots/deadbeef --served-model-name", done.stderr,
                      "the second attempt still passed the repo id")

    def test_a_throttled_hub_is_not_reported_as_a_memory_problem(self):
        """Two failures in one: reporting success on a server that never came
        up, and blaming the KV cache for the neighbours' traffic. The first
        would send the harness at a dead port; the second would send the
        operator to lower --max-model-len, which changes nothing."""
        done = self._drive_start_server(THROTTLED_VLLM, model_on_disk=False, attempts=2)
        self.assertNotEqual(done.returncode, 0,
                            "start_server reported success without a server")
        self.assertIn("429", done.stderr)
        self.assertNotIn("te weinig geheugen voor de KV-cache", done.stderr)


# A vLLM that dies the way a multi-card start dies: the reason is in the
# worker's own lines, and the API server -- whose traceback is the tail of the
# log, and the only part anyone sees -- says only that a background process
# went away. Taken from the 08:13 log of a two-card start.
WORKER_FAILURE = """
cat <<'VLLMEOF'
(VllmWorker rank=1 pid=1301) INFO 09-10 08:13:50 [multiproc_executor.py:558] Worker ready
(VllmWorker rank=1 pid=1301) ERROR 09-10 08:13:52 [multiproc_executor.py:600] Traceback (most recent call last):
(VllmWorker rank=1 pid=1301) ERROR 09-10 08:13:52 [multiproc_executor.py:600]   File "/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py", line 209, in __init__
(VllmWorker rank=1 pid=1301) ERROR 09-10 08:13:52 [multiproc_executor.py:600]     self.shared_memory = shared_memory.SharedMemory(create=True, size=self.total_bytes)
(VllmWorker rank=1 pid=1301) ERROR 09-10 08:13:52 [multiproc_executor.py:600]                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorker rank=1 pid=1301) ERROR 09-10 08:13:52 [multiproc_executor.py:600] OSError: [Errno 28] No space left on device
(EngineCore pid=1281) ERROR 09-10 08:13:54 [core.py:1374]     return func(*args, **kwargs)
(EngineCore pid=1281) ERROR 09-10 08:13:54 [core.py:1374]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1281) ERROR 09-10 08:13:54 [core.py:1374]   File "/dist-packages/vllm/v1/executor/abstract.py", line 110, in __init__
(EngineCore pid=1281) ERROR 09-10 08:13:54 [core.py:1374]     self._init_executor()
(EngineCore pid=1281) ERROR 09-10 08:13:54 [core.py:1374] Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.
(APIServer pid=1000)   File "/dist-packages/vllm/v1/engine/utils.py", line 1320, in wait_for_engine_startup
(APIServer pid=1000)     raise RuntimeError(
(APIServer pid=1000) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
VLLMEOF
exit 1
"""

# One card, one process, and a reason the log states outright.
SINGLE_CARD_OOM = """
echo "$ARGS_ECHO"
echo "(EngineCore pid=1281) ERROR 09-10 08:13:54 [core.py:1374] ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization."
exit 1
"""


class TestAWorkerThatDiedInAnotherProcess(StartServerHarness):
    """With --tensor-parallel-size above 1 vLLM runs a process per card, and
    the traceback at the end of the log is the API server's: it says a
    background process went away and refers to a stack trace that is hundreds
    of lines further up, in the worker's own output. The wrapper has to fetch
    that, and it has to stop reporting stack frames as the suspected cause --
    SERVER_ERROR_PATTERNS ends in a bare "Error", which matched the ERROR
    prefix of every frame and drowned the one line that named a reason."""

    def _fail(self, tensor_parallel: int = 2, body: str = WORKER_FAILURE):
        done = self._drive_start_server(fake_vllm(body), attempts=2,
                                        tensor_parallel=tensor_parallel)
        self.assertEqual(done.returncode, 1, done.stderr)
        return done

    @staticmethod
    def _section(stderr: str, heading: str) -> str:
        rest = stderr.split(heading, 1)[1]
        return rest.split("\n--- ", 1)[0]

    def test_the_suspected_cause_names_a_cause_and_not_a_stack_frame(self):
        done = self._fail()
        self.assertIn("vermoedelijke oorzaak", done.stderr)
        suspects = self._section(done.stderr, "vermoedelijke oorzaak")
        self.assertIn("OSError: [Errno 28] No space left on device", suspects)
        self.assertNotIn("return func(*args, **kwargs)", suspects)
        self.assertNotIn("self._init_executor()", suspects)
        self.assertNotIn('File "', suspects)

    def test_the_workers_own_lines_are_surfaced(self):
        done = self._fail()
        self.assertIn("wat de workers zelf zeiden", done.stderr)
        workers = self._section(done.stderr, "wat de workers zelf zeiden")
        self.assertIn("OSError: [Errno 28] No space left on device", workers)
        self.assertNotIn("APIServer", workers)

    def test_the_diagnosis_names_the_three_things_that_break_here(self):
        done = self._fail()
        for expected in ("shm-size", "NCCL", "TENSOR_PARALLEL=1"):
            self.assertIn(expected, done.stderr,
                          f"the multi-card diagnosis should mention {expected}")

    def test_it_does_not_read_as_a_kv_cache_problem(self):
        """The generic message sends the operator after --max-model-len and
        GPU_UTIL. Neither has anything to do with a worker that never got to
        the point of allocating a cache."""
        done = self._fail()
        self.assertNotIn("verlaag --max-model-len", done.stderr)

    def test_one_card_keeps_the_message_it_had(self):
        done = self._fail(tensor_parallel=1, body=SINGLE_CARD_OOM)
        self.assertIn("verlaag --max-model-len", done.stderr)
        self.assertNotIn("shm-size", done.stderr)
        suspects = self._section(done.stderr, "vermoedelijke oorzaak")
        self.assertIn("No available memory for the cache blocks", suspects)

    def test_a_worker_failure_is_not_mistaken_for_a_flashinfer_one(self):
        """Retrying without the sampler, or on another attention backend,
        costs a start-up and measures nothing different."""
        done = self._fail()
        self.assertEqual(len(self.starts(done)), 1, done.stderr)
        self.assertIn("--tensor-parallel-size 2", self.starts(done)[0])


# The pod as it is on two 5090s: NCCL wants the direct PCIe path between the
# cards, does not get it, and the worker dies opening the communicator.
NCCL_P2P_FAILURE = """
cat <<'VLLMEOF'
(VllmWorker rank=1 pid=1301) ERROR 09-10 08:13:52 [multiproc_executor.py:600]   File "/dist-packages/torch/distributed/distributed_c10d.py", line 1600, in init
(VllmWorker rank=1 pid=1301) ERROR 09-10 08:13:52 [multiproc_executor.py:600] torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:1970, unhandled system error
(EngineCore pid=1281) ERROR 09-10 08:13:54 [core.py:1374] Exception: WorkerProc initialization failed due to an exception in a background process.
(APIServer pid=1000) RuntimeError: Engine core initialization failed. Failed core proc(s): {}
VLLMEOF
exit 1
"""

NEEDS_NO_P2P = fake_vllm(f"""
    if [ "${{NCCL_P2P_DISABLE:-}}" = 1 ]; then {STARTED}; fi
    {NCCL_P2P_FAILURE}""")


class TestCardsThatCannotReachEachOther(StartServerHarness):
    """Two consumer cards in one pod do not always have the PCIe peer-to-peer
    path NCCL reaches for first, and NCCL dies in the worker rather than
    routing around it. Going over host memory instead costs throughput on
    every exchange between the cards, so the wrapper may only do it when the
    log actually blames NCCL -- and has to say that it did."""

    def test_the_run_is_retried_over_host_memory(self):
        done = self._drive_start_server(NEEDS_NO_P2P, attempts=3, tensor_parallel=2)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(len(self.starts(done)), 2, done.stderr)
        self.assertIn("[ok] NCCL_P2P_OFF=1", done.stdout)
        self.assertIn("NCCL_P2P_DISABLE=1", done.stderr)

    def test_the_operator_is_told_it_costs_speed(self):
        done = self._drive_start_server(NEEDS_NO_P2P, attempts=3, tensor_parallel=2)
        self.assertIn("trager", done.stderr)
        self.assertIn("noteer het", done.stderr)

    def test_it_is_recorded_next_to_the_other_sticky_choices(self):
        done = self._drive_start_server(NEEDS_NO_P2P, attempts=3, tensor_parallel=2)
        self.assertEqual(done.returncode, 0, done.stderr)
        state = os.path.join(self.tmp, "state", "current_nccl_p2p")
        with open(state, encoding="utf-8") as handle:
            self.assertEqual(handle.read().strip(), "1")

    def test_one_card_never_takes_this_branch(self):
        """A single-card run has no communicator to fail, and the same log
        lines there mean something else entirely."""
        done = self._drive_start_server(fake_vllm(NCCL_P2P_FAILURE), attempts=3,
                                        tensor_parallel=1)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertEqual(len(self.starts(done)), 1, done.stderr)
        self.assertNotIn("NCCL_P2P_DISABLE=1", done.stderr)

    def test_a_healthy_start_that_merely_mentions_p2p_is_left_alone(self):
        """vLLM says "custom allreduce is disabled because your platform lacks
        GPU P2P capability" on starts that are entirely fine."""
        chatty = fake_vllm(f"""
            echo "INFO [custom_all_reduce.py:98] custom allreduce is disabled because your platform lacks GPU P2P capability"
            {STARTED}""")
        done = self._drive_start_server(chatty, attempts=3, tensor_parallel=2)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(len(self.starts(done)), 1, done.stderr)
        self.assertIn("[ok] NCCL_P2P_OFF=0", done.stdout)


# What an image whose torch is built against CUDA 13 does on a pod with a
# 570 driver: every worker dies at CUDA initialisation, and vLLM reports that
# as a background process that went away. The real line is in the worker's own
# output, and it names neither vLLM nor the flag that was used.
DRIVER_TOO_OLD = """
cat <<'VLLMEOF'
(Worker pid=1455) ERROR 09-10 08:13:53 [multiproc_executor.py:944] WorkerProc failed to start.
(Worker pid=1455) ERROR 09-10 08:13:53 [multiproc_executor.py:944]   File "/dist-packages/vllm/v1/worker/gpu_worker.py", line 412, in init_device
(Worker pid=1455) ERROR 09-10 08:13:53 [multiproc_executor.py:944]     torch._C._cuda_init()
(Worker pid=1455) ERROR 09-10 08:13:53 [multiproc_executor.py:944] RuntimeError: The NVIDIA driver on your system is too old (found version 12080). Please update your GPU driver.
(EngineCore pid=1281) ERROR 09-10 08:13:54 [core.py:1374] Exception: WorkerProc initialization failed due to an exception in a background process.
(APIServer pid=1000) RuntimeError: Engine core initialization failed. Failed core proc(s): {}
VLLMEOF
exit 1
"""


class TestATorchThatCannotOpenTheCard(StartServerHarness):
    """The measurement of 10 September: two 5090s, driver 570.144, an image
    whose torch wants CUDA 13. Every worker died in torch._C._cuda_init(), and
    because --tensor-parallel-size was 2 the failure arrived dressed as a
    worker that went away -- a machine problem, apparently, on a machine that
    was fine. No flag helps against this one, so nothing may be retried and
    nothing may be blamed on /dev/shm or on NCCL."""

    def test_the_driver_is_named_and_not_the_shared_memory(self):
        done = self._drive_start_server(fake_vllm(DRIVER_TOO_OLD), attempts=3,
                                        tensor_parallel=2)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("driver", done.stderr)
        self.assertIn("torch", done.stderr)
        self.assertNotIn("shm-size", done.stderr)
        self.assertNotIn("NCCL_P2P_DISABLE", done.stderr)

    def test_nothing_is_retried_because_nothing_would_help(self):
        done = self._drive_start_server(fake_vllm(DRIVER_TOO_OLD), attempts=3,
                                        tensor_parallel=2)
        self.assertEqual(len(self.starts(done)), 1, done.stderr)

    def test_it_does_not_read_as_a_kv_cache_problem_either(self):
        done = self._drive_start_server(fake_vllm(DRIVER_TOO_OLD), attempts=3,
                                        tensor_parallel=1)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertNotIn("verlaag --max-model-len", done.stderr)
        self.assertIn("ook TENSOR_PARALLEL=1 niet", done.stderr,
                      "a single card hides this failure, it does not fix it")

    def test_the_worker_line_reaches_the_operator(self):
        done = self._drive_start_server(fake_vllm(DRIVER_TOO_OLD), attempts=3,
                                        tensor_parallel=2)
        self.assertIn("The NVIDIA driver on your system is too old", done.stderr)


class TestTorchIsAskedBeforeTheModelIsDownloaded(unittest.TestCase):
    """Fifteen minutes of loading to find out that torch and the driver do not
    agree about CUDA is fifteen minutes of rent. torch answers the same
    question in a second, before anything has been fetched or started."""

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _probe(self, answer: str, status: int) -> subprocess.CompletedProcess:
        py = os.path.join(self.tmp, "fake-python")
        with open(py, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\n"
                         f"printf '%s' {shell_quote(answer)}\n"
                         f"exit {status}\n")
        os.chmod(py, os.stat(py).st_mode | stat.S_IEXEC)
        program = "\n".join([
            "set -Eeuo pipefail",
            'say() { echo "[say] $*" >&2; }',
            'warn() { echo "[warn] $*" >&2; }',
            'die() { echo "[die] $*" >&2; exit 1; }',
            f'MOCK=0; PY="{py}"',
            function_body("driver_cuda_version"),
            function_body("check_driver_matches_torch"),
            "check_driver_matches_torch",
            "echo reached-the-end",
        ])
        return subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    def test_a_driver_that_is_too_old_stops_the_run_there(self):
        done = self._probe("13.0|The NVIDIA driver on your system is too old "
                           "(found version 12080).", status=1)
        self.assertEqual(done.returncode, 1)
        self.assertIn("[die]", done.stderr)
        self.assertIn("CUDA 13.0", done.stderr)
        self.assertIn("cu128", done.stderr, "the way out belongs in the message")
        self.assertNotIn("reached-the-end", done.stdout)

    def test_a_healthy_pod_says_so_and_carries_on(self):
        done = self._probe("12.8|ok", status=0)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("reached-the-end", done.stdout)
        self.assertIn("CUDA 12.8", done.stderr)

    def test_a_pod_without_torch_yet_is_not_an_error(self):
        """preflight runs before vLLM is installed on a clean pod; cmd_setup
        asks again once it is."""
        done = self._probe("", status=3)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("reached-the-end", done.stdout)
        self.assertNotIn("[die]", done.stderr)
        self.assertNotIn("[warn]", done.stderr)

    def test_any_other_refusal_warns_instead_of_guessing(self):
        done = self._probe("12.8|CUDA unknown error", status=1)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[warn]", done.stderr)
        self.assertIn("CUDA unknown error", done.stderr)

    def test_the_probe_runs_in_preflight_and_after_the_install(self):
        self.assertIn("check_driver_matches_torch", function_body("preflight"))
        self.assertIn("check_driver_matches_torch", function_body("cmd_setup"))


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


class TestWriteAccessIsProven(unittest.TestCase):
    """`git ls-remote` succeeds with a read-only token. The whole point of
    asking before the measurement is to catch the token that cannot push, so
    the check has to push."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.origin = os.path.join(self.tmp, "origin.git")
        subprocess.run(["git", "init", "--bare", "-q", self.origin], check=True)
        self.repo = os.path.join(self.tmp, "repo")
        subprocess.run(["git", "init", "-q", "-b", "main", self.repo], check=True)
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        with open(os.path.join(self.repo, "a.txt"), "w", encoding="utf-8") as handle:
            handle.write("x\n")
        subprocess.run(["git", "-C", self.repo, "add", "a.txt"], check=True)
        subprocess.run(["git", "-C", self.repo, "commit", "-qm", "init"], check=True,
                       env=env)
        subprocess.run(["git", "-C", self.repo, "remote", "add", "origin", self.origin],
                       check=True)

    def _refuse_pushes(self):
        """A remote that reads fine and refuses every write: branch protection,
        a read-only token, a revoked scope -- all the same from here."""
        hook = os.path.join(self.origin, "hooks", "pre-receive")
        with open(hook, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\nexit 1\n")
        os.chmod(hook, 0o755)

    def _run(self):
        text = script_text()
        bodies = []
        for name in ("setup_git_credentials", "_push_access_hint", "check_push_access"):
            found = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.M | re.S)
            self.assertIsNotNone(found, f"{name} is gone")
            bodies.append(found.group(0))
        script = ("set -Eeuo pipefail\n"
                  f'REPO_DIR="{self.repo}"\n'
                  f'STATE_DIR="{self.tmp}/state"\n'
                  "PUSH_RESULTS=1\nMOCK=0\n"
                  'say()   { echo "$*"; }\n'
                  'warn()  { echo "LET OP: $*" >&2; }\n'
                  'head_() { echo "$*"; }\n'
                  + "\n".join(bodies) + "\ncheck_push_access\n")
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                              timeout=120)

    def _remote_branches(self):
        listing = subprocess.run(["git", "--git-dir", self.origin, "for-each-ref",
                                  "--format=%(refname:short)", "refs/heads"],
                                 capture_output=True, text=True, check=True).stdout
        return listing.split()

    def test_a_writable_remote_passes(self):
        done = self._run()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("push-toegang in orde", done.stdout)

    def test_the_probe_branch_does_not_stay_behind(self):
        self.assertEqual(self._run().returncode, 0)
        self.assertEqual(self._remote_branches(), [],
                         "de proefbranch moet weer weg zijn")

    def test_a_remote_that_only_allows_reading_is_refused(self):
        """This is the case the check exists for, and the one `ls-remote`
        used to wave through -- six hours before the push actually failed."""
        self._refuse_pushes()
        listing = subprocess.run(["git", "-C", self.repo, "ls-remote", "origin"],
                                 capture_output=True, text=True)
        self.assertEqual(listing.returncode, 0,
                         "lezen moet lukken, anders test dit iets anders")
        done = self._run()
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertNotIn("push-toegang in orde", done.stdout)
        self.assertIn("niet naartoe pushen", done.stderr)
        self.assertIn("GITHUB_TOKEN", done.stderr, "de aanwijzing hoort erbij")

    def test_an_unreachable_remote_gets_its_own_message(self):
        """'geen netwerk' en 'wel netwerk, geen schrijfrechten' vragen om een
        andere oplossing, dus ze horen niet hetzelfde te melden."""
        subprocess.run(["git", "-C", self.repo, "remote", "set-url", "origin",
                        os.path.join(self.tmp, "bestaat-niet.git")], check=True)
        done = self._run()
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("niet bereiken", done.stderr)


class TestResumingAnOlderDirectory(unittest.TestCase):
    """Resuming is what makes a six-hour measurement survivable. Resuming into
    a directory measured with a different behaviour model is how a run finishes
    in ten minutes having measured nothing."""

    def test_the_harness_decides_what_is_pending(self):
        """The check that a run id is not a description of the setup lives in
        the harness, so it also applies to a matrix run started by hand."""
        body = function_body("pending_ids")
        self.assertIn("-m stresstest pending", body)
        self.assertNotIn("os.path.exists", body,
                         "het tellen hoort niet meer in de shell te zitten")
        self.assertIn("RESUME_ANYWAY", body)

    def test_a_group_stops_when_the_setup_moved(self):
        run_group = function_body("run_group")
        self.assertIn("|| die", run_group.split("if [ -z")[0],
                      "een afwijkende opstelling moet de reeks stoppen, niet stil "
                      "doorgaan alsof alles al gemeten is")

    def test_the_engine_group_asks_the_same_question(self):
        """The engine variants used to decide for themselves whether a run was
        already there, on the same name-only evidence."""
        body = function_body("run_engine_group")
        self.assertIn("pending_ids engine", body)
        self.assertNotIn('[ -f "$dir/runs/engine_$name/run.json" ]', body,
                         "ook hier zegt het bestaan van een run.json niets over "
                         "waarmee die run gemeten is")

    def test_resume_anyway_exists_and_is_off_by_default(self):
        text = script_text()
        self.assertRegex(text, r"(?m)^RESUME_ANYWAY=0$")
        self.assertIn("--resume-anyway) RESUME_ANYWAY=1", text)
        self.assertIn("--resume-anyway", function_body("usage"))

    def test_the_resumed_directory_is_described_not_just_named(self):
        """`hervat in results/20260908-...` was true and useless. How old it is
        and how much is already in it is what makes a wrong one visible."""
        self.assertIn("describe_results_dir", function_body("cmd_all"))
        described = function_body("describe_results_dir")
        self.assertIn("aangemaakt", described)
        self.assertIn("runs", described)


class TestSkippedLessonDoesNotLeaveAStaleReport(unittest.TestCase):
    """RESULTATEN.md in the repository root survives between runs. Pushing it
    along with a measurement that did not write it claims a coverage that
    measurement does not have."""

    def test_the_combined_report_is_only_pushed_when_this_run_wrote_it(self):
        body = function_body("push_results")
        self.assertIn('REPORT_WRITTEN', body)
        self.assertNotIn("[ -f RESULTATEN.md ] && git add -f RESULTATEN.md", body,
                         "onvoorwaardelijk meestagen is precies de fout")

    def test_skipping_phase_two_only_reuses_a_matching_lesson(self):
        body = function_body("reusable_lesson_dir")
        self.assertIn("fingerprint --same", body,
                      "een oudere lesmap mag alleen mee als de opstelling klopt")
        self.assertIn("last_lesson_dir", body)

    def test_a_run_without_phase_two_says_what_it_did_not_update(self):
        body = function_body("cmd_all")
        self.assertIn("reusable_lesson_dir", body)
        self.assertIn("REPORT_WRITTEN=1", body)
        tail = body.split("reusable_lesson_dir")[1]
        self.assertIn("niet bijgewerkt", tail)
        self.assertIn("--also", tail, "de melding hoort het commando mee te geven "
                                      "dat het rapport alsnog bijwerkt")


class TestTheMetricsGate(unittest.TestCase):
    """The gate that decides whether a measurement may start.

    It refused two rented pods within a minute of the model being loaded, both
    times over "KV-bezetting" -- a series that was in the body it had just
    scraped, and that the harness read without trouble on the very same server.
    The refusal was in the plumbing: `printf "%s" "$body" | grep -q` lets grep
    stop at its first match while printf is still writing, printf dies of
    SIGPIPE, and `set -o pipefail` turns that into "the series is not there".
    A series near the top of a body larger than the pipe buffer therefore reads
    as missing, and vLLM prints the KV gauge near the top.

    The bodies below are built the same way round: KV first, tens of thousands
    of histogram buckets after it, preemptions last."""

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def body(self, *, kv: bool = True, preemptions: bool = True,
             prefix: bool = True, buckets: int = 4000) -> str:
        lines = ["# HELP vllm:kv_cache_usage_perc KV cache usage",
                 "# TYPE vllm:kv_cache_usage_perc gauge"]
        if kv:
            lines.append('vllm:kv_cache_usage_perc{model_name="qwen3-coder"} 0.0274')
        for i in range(buckets):
            lines.append('vllm:request_latency_seconds_bucket'
                         f'{{model_name="qwen3-coder",le="{i}.0"}} {i}')
        if prefix:
            lines.append('vllm:prefix_cache_queries_total{model_name="qwen3-coder"} 54495.0')
            lines.append('vllm:prefix_cache_hits_total{model_name="qwen3-coder"} 18112.0')
        if preemptions:
            lines.append('vllm:num_preemptions_total{model_name="qwen3-coder"} 0.0')
        return "\n".join(lines) + "\n"

    def write(self, body: str, name: str = "metrics.txt") -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        return path

    def missing(self, body: str) -> str:
        program = "\n".join([
            "set -Eeuo pipefail",
            function_body("missing_metrics"),
            'missing_metrics "$(cat "$1")"',
        ])
        done = subprocess.run(["bash", "-c", program, "pod.sh", self.write(body)],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.strip()

    def test_a_body_larger_than_the_pipe_buffer_is_read_correctly(self):
        body = self.body()
        self.assertGreater(len(body), 65536, "this body would not have shown the bug")
        self.assertEqual(self.missing(body), "")

    def test_the_same_body_small_enough_to_fit_still_passes(self):
        self.assertEqual(self.missing(self.body(buckets=5)), "")

    def test_a_series_that_really_is_gone_is_still_reported(self):
        self.assertEqual(self.missing(self.body(kv=False)), "KV-bezetting")
        self.assertEqual(self.missing(self.body(preemptions=False)), "preempties")
        self.assertEqual(self.missing(self.body(prefix=False)), "prefix-cache")
        self.assertEqual(self.missing(self.body(kv=False, preemptions=False, prefix=False)),
                         "preempties prefix-cache KV-bezetting")

    def test_the_older_name_for_the_kv_gauge_is_accepted(self):
        """vLLM renamed gpu_cache_usage_perc to kv_cache_usage_perc. The
        harness reads both, and so must the gate."""
        body = self.body(kv=False).replace(
            "# TYPE vllm:kv_cache_usage_perc gauge",
            '# TYPE vllm:gpu_cache_usage_perc gauge\n'
            'vllm:gpu_cache_usage_perc{model_name="qwen3-coder"} 0.0274')
        self.assertEqual(self.missing(body), "")

    def _gate(self, bodies: list[str]) -> tuple[subprocess.CompletedProcess, int]:
        """check_metrics against a server that answers with `bodies` in turn."""
        counter = os.path.join(self.tmp, "attempts")
        with open(counter, "w", encoding="utf-8") as handle:
            handle.write("0")
        paths = [self.write(body, f"body{i}.txt") for i, body in enumerate(bodies)]
        program = "\n".join([
            "set -Eeuo pipefail",
            'say() { echo "[say] $*" >&2; }',
            'warn() { echo "[warn] $*" >&2; }',
            'die() { echo "[die] $*" >&2; exit 1; }',
            "sleep() { :; }",              # the retries wait five seconds each
            "warm_up_server() { :; }",
            "FORCE=0; PORT=8000",
            'COUNTER="$1"; shift; BODIES=("$@")',
            # Stands in for the scrape: hands out one body per attempt, and the
            # last one for every attempt after that.
            'curl() {',
            '  local n; n=$(( $(cat "$COUNTER") + 1 )); echo "$n" > "$COUNTER"',
            '  [ "$n" -le "${#BODIES[@]}" ] || n="${#BODIES[@]}"',
            '  cat "${BODIES[$((n - 1))]}"',
            "}",
            function_body("missing_metrics"),
            function_body("show_metric_candidates"),
            function_body("check_metrics"),
            "check_metrics",
            "echo reached-the-end",
        ])
        done = subprocess.run(["bash", "-c", program, "pod.sh", counter, *paths],
                              capture_output=True, text=True)
        with open(counter, encoding="utf-8") as handle:
            return done, int(handle.read())

    def test_a_gauge_that_arrives_late_gets_the_retries_it_is_refused_over(self):
        """The KV gauge is not written before the engine has scheduled
        something. The loop used to stop as soon as the preemption counter was
        there -- the series that arrives first -- and then refused over the one
        it had just given up waiting for."""
        done, attempts = self._gate([self.body(kv=False), self.body(kv=False), self.body()])
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("reached-the-end", done.stdout)
        self.assertEqual(attempts, 3, "the gate did not spend its retries")
        self.assertNotIn("[die]", done.stderr)

    def test_a_gate_that_stops_a_run_says_what_it_did_see(self):
        """Another start-up is a quarter of an hour of rent. The names it found
        turn the next round into a read instead of a rerun."""
        done, _ = self._gate([self.body(kv=False)])
        self.assertEqual(done.returncode, 1)
        self.assertIn("[die]", done.stderr)
        self.assertIn("KV-bezetting", done.stderr)
        self.assertIn("vllm:num_preemptions_total", done.stderr)
        self.assertIn("vllm:prefix_cache_hits_total", done.stderr)


class TestAFailedRunDoesNotKeepThePodRunning(unittest.TestCase):
    """What a refused start actually cost: not the run, but the hours the pod
    stood idle afterwards with a deadman still eight hours out."""

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _shorten(self, deadman_at: str | None, armed: bool = True,
                 hours: str = "0.5") -> subprocess.CompletedProcess:
        if armed:
            with open(os.path.join(self.tmp, "deadman.pid"), "w", encoding="utf-8") as handle:
                handle.write("424242")
        if deadman_at is not None:
            with open(os.path.join(self.tmp, "deadman_at"), "w", encoding="utf-8") as handle:
                handle.write(deadman_at + "\n")
        program = "\n".join([
            "set -Eeuo pipefail",
            'warn() { echo "[warn] $*" >&2; }',
            'arm_deadman() { echo "[arm] $1" >&2; }',
            f'STATE_DIR="{self.tmp}"; DEADMAN_PID_FILE="{self.tmp}/deadman.pid"',
            function_body("shorten_deadman"),
            f'shorten_deadman {hours}',
            "echo reached-the-end",
        ])
        return subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    def when(self, seconds: int) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + seconds))

    def test_a_deadman_hours_out_is_brought_forward(self):
        done = self._shorten(self.when(8 * 3600))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[arm] 0.5", done.stderr)

    def test_a_deadman_that_is_already_closer_is_left_alone(self):
        """The ten minutes 'all' arms after a successful push must not be
        stretched to half an hour by a failure in the push that follows it."""
        done = self._shorten(self.when(600))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("[arm]", done.stderr)

    def test_without_a_deadman_nothing_is_armed(self):
        """No deadman is the operator saying the machine stops when they say
        so. An error is not the moment to overrule that."""
        done = self._shorten(self.when(8 * 3600), armed=False)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("[arm]", done.stderr)

    def test_a_time_it_cannot_read_leaves_the_deadman_standing(self):
        done = self._shorten("over 8.1u")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("reached-the-end", done.stdout)
        self.assertNotIn("[arm]", done.stderr)

    def test_the_fatal_path_goes_through_it(self):
        die = re.search(r"^die\(\).*$", script_text(), re.M)
        self.assertIsNotNone(die)
        self.assertIn("shorten_deadman", die.group(0))
        self.assertIn('ABORT_GRACE_HOURS="${ABORT_GRACE_HOURS:-', script_text())


class TestRescuingAMeasurementThatWasLeftBehind(unittest.TestCase):
    """`all` pushes at the end. Everything that stops earlier leaves the runs it
    did finish on a rented disk, and the only way out used to be an scp over a
    port from the dashboard -- see VALKUILEN.md on how well that goes."""

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.trace = os.path.join(self.tmp, "trace")
        open(self.trace, "w", encoding="utf-8").close()

    def _push(self, latest: str = "", lesson: str = "", *args: str, report_fails: bool = False):
        program = "\n".join([
            "set -Eeuo pipefail",
            'say() { echo "[say] $*" >&2; }',
            'warn() { echo "[warn] $*" >&2; }',
            'die() { echo "[die] $*" >&2; exit 1; }',
            "detect_gpu() { GPU_NAME=test; }",
            'describe_results_dir() { echo "3 runs"; }',
            'push_results() { echo "[push] $*" >&2; echo "push $*" >> "$TRACE"; }',
            ('stresstestreport() { echo "report $*" >> "$TRACE"; return 1; }' if report_fails
             else 'stresstestreport() { echo "report $*" >> "$TRACE"; }'),
            "PY=stresstestreport",
            f'TRACE="{self.trace}"',
            f'latest_matrix_dir() {{ printf "%s" "{latest}"; }}',
            f'STATE_DIR="{self.tmp}"',
            function_body("cmd_push"),
            "cmd_push " + " ".join(args),
            "echo reached-the-end",
        ])
        if lesson:
            with open(os.path.join(self.tmp, "last_lesson_dir"), "w", encoding="utf-8") as handle:
                handle.write(lesson + "\n")
        return subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    def test_without_arguments_it_pushes_the_last_measurement(self):
        done = self._push(latest="results/20260910-084827_matrix")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[push] results/20260910-084827_matrix", done.stderr)

    def test_a_lesson_that_belongs_to_it_goes_along(self):
        lesson = os.path.join(self.tmp, "les")
        os.makedirs(lesson)
        done = self._push(latest="results/20260910-084827_matrix", lesson=lesson)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(f"[push] results/20260910-084827_matrix {lesson}", done.stderr)

    def test_a_lesson_directory_that_is_gone_is_not_pushed(self):
        done = self._push(latest="results/x_matrix", lesson=os.path.join(self.tmp, "weg"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[push] results/x_matrix\n", done.stderr)

    def test_an_explicit_directory_is_taken_as_given(self):
        done = self._push("", "", "results/een", "results/twee")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[push] results/een results/twee", done.stderr)

    def trace_lines(self) -> list[str]:
        with open(self.trace, encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def test_the_summary_is_rebuilt_before_it_is_pushed(self):
        """summary.csv and summary.json are written from the memory of the
        measuring session. A run that is interrupted pushes a summary of the
        group it happened to be in -- the first rescue push carried 1 of 37
        runs. Rebuilding from runs/*/run.json needs no GPU."""
        done = self._push("", "", self.tmp)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.trace_lines(),
                         [f"report -m stresstest report {self.tmp}", f"push {self.tmp}"],
                         "the summary has to be rebuilt, and before it is pushed")

    def test_a_summary_that_cannot_be_rebuilt_does_not_cost_the_data(self):
        """Whatever is wrong with the report, the runs are what must not stay
        behind on a rented disk."""
        done = self._push("", "", self.tmp, report_fails=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[warn]", done.stderr)
        self.assertIn(f"push {self.tmp}", self.trace_lines())

    def test_nothing_to_push_says_so_instead_of_pushing_nothing(self):
        done = self._push(latest="")
        self.assertEqual(done.returncode, 1)
        self.assertIn("[die]", done.stderr)


class TestHowMuchMemoryTheCardHas(unittest.TestCase):
    """VRAM_GB is what every gigabyte in the report is computed from: the KV
    percentages are multiplied by it, and vraag 3 -- past het ook op 72 GB? --
    is nothing else.

    On a MIG instance `nvidia-smi --query-gpu=memory.total` answers "[N/A]".
    `awk 'BEGIN{printf m/1024}'` makes that a silent 0.0, which is not empty,
    so the 96.0 fallback never fired. A whole matrix was measured that way: the
    conclusion opened with "past een klas van 20 op 0 GB" and marked every
    alternative card as fitting."""

    def setUp(self):
        if not shutil.which("bash"):
            self.skipTest("no bash available")

    def _detect(self, memory_total: str, torch_gb: str | None = None,
                vram_gb: str = "", cards: int = 1, tensor_parallel: int = 1):
        torch = (f'faketorch() {{ printf "%s\\n" "{torch_gb}"; }}' if torch_gb
                 else "faketorch() { return 1; }")
        program = "\n".join([
            "set -Eeuo pipefail",
            'say() { echo "[say] $*" >&2; }',
            'warn() { echo "[warn] $*" >&2; }',
            'die() { echo "[die] $*" >&2; exit 1; }',
            torch,
            "PY=faketorch; MOCK=0",
            f"TENSOR_PARALLEL={tensor_parallel}",
            (f'VRAM_GB="{vram_gb}"' if vram_gb else ":"),
            # command -v finds shell functions, so this stands in for the tool.
            'nvidia-smi() {',
            '  case "$*" in',
            f'    *name,memory.total*) for _ in $(seq 1 {cards}); do echo "NVIDIA RTX PRO 6000 Blackwell Server Edition, {memory_total}"; done ;;',
            f'    *--query-gpu=name\ *|*--query-gpu=name) for _ in $(seq 1 {cards}); do echo "NVIDIA RTX PRO 6000"; done ;;',
            '    *power.default_limit*) echo "600.00" ;;',
            '    *) echo "" ;;',
            "  esac",
            "}",
            function_body("gpu_memory_gb_from_torch"),
            function_body("detect_gpu"),
            "detect_gpu",
            'echo "VRAM_GB=$VRAM_GB SOURCE=$VRAM_SOURCE COUNT=$GPU_COUNT"',
        ])
        return subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    def test_a_card_that_answers_normally(self):
        done = self._detect("98304")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("VRAM_GB=96.0 SOURCE=nvidia-smi", done.stdout)

    def test_a_mig_slice_is_asked_of_torch_instead(self):
        """torch reports the slice, which is exactly the pool vLLM divides."""
        done = self._detect("[N/A]", torch_gb="47.5")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("VRAM_GB=47.5 SOURCE=torch", done.stdout)

    def test_a_container_that_may_not_ask_is_the_same_case(self):
        """What a RunPod container holding one MIG slice actually answers:
        not "[N/A]" but "[Insufficient Permissions]". Anything that is not a
        plain number is no answer."""
        done = self._detect("[Insufficient Permissions]", torch_gb="47.4")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("VRAM_GB=47.4 SOURCE=torch", done.stdout)

    def test_a_mig_slice_without_torch_is_marked_as_a_guess(self):
        """The number it falls back on is the one from the budget request. It
        may not pass for something the machine said."""
        done = self._detect("[N/A]")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("SOURCE=aanname", done.stdout)
        self.assertNotIn("VRAM_GB=0", done.stdout)

    def test_a_number_from_the_operator_wins_from_both(self):
        done = self._detect("98304", torch_gb="47.5", vram_gb="48")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("VRAM_GB=48 SOURCE=opgegeven", done.stdout)

    def test_two_cards_are_added_up(self):
        done = self._detect("32768", cards=2, tensor_parallel=2)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("VRAM_GB=64.0 SOURCE=nvidia-smi", done.stdout)

    def test_a_resume_is_checked_before_the_server_starts(self):
        """A directory left on a shared network volume by another card cannot
        be measured into. Discovering that at the first group means the model
        is already loaded -- a quarter of an hour of rent for an answer the
        configuration alone could give."""
        body = function_body("cmd_all")
        resume = body.index("hervat in bestaande map")
        check = body.index("pending_ids rampup")
        server = body.index("start_server")
        self.assertLess(resume, check, "the check has to follow the directory it checks")
        self.assertLess(check, server, "the check has to come before vLLM is started")

    def test_preflight_refuses_to_measure_on_a_guess(self):
        body = function_body("preflight")
        self.assertIn('VRAM_SOURCE" = "aanname"', body)
        self.assertIn("FORCE", body)
        self.assertIn("VRAM_GB=48", body, "the refusal has to say how to supply the number")
        self.assertIn("bron: $VRAM_SOURCE", body,
                      "the pool line has to say where the number came from")


if __name__ == "__main__":
    unittest.main()
