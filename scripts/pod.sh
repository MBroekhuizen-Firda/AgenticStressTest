#!/usr/bin/env bash
#
# Unattended driver for the agentic-coding stress test on a rented GPU pod.
#
# Everything the README describes by hand -- installing vLLM, fetching the
# model, starting the server, running phase 1, restarting the server for each
# engine variant, running phase 2, writing the conclusion -- happens here in
# one command. It is resumable: runs already on disk are skipped, so a dropped
# SSH session costs you the current run and nothing else.
#
# Operator-facing output is Dutch, like the report; identifiers and comments
# are English, like the rest of the harness.
#
#   scripts/pod.sh all              de hele test, onbewaakt
#   scripts/pod.sh setup            alleen installeren en het model ophalen
#   scripts/pod.sh serve            alleen vLLM starten
#   scripts/pod.sh --help           alle commando's
#
set -Eeuo pipefail

# --------------------------------------------------------------------------
# Knobs. Every one of them can be overridden from the environment:
#   VRAM_GB=72 scripts/pod.sh all
# --------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-$(dirname "$REPO_DIR")}"

MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
SERVED_NAME="${SERVED_NAME:-qwen3-coder}"
PORT="${PORT:-8000}"
HOST_BIND="${HOST_BIND:-0.0.0.0}"

# The three flags the engine variants sweep. These are the baseline values;
# each variant replaces all three wholesale.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

GPU_UTIL="${GPU_UTIL:-0.90}"
# Which attention backend vLLM uses. Empty means "let vLLM choose", which is
# right until its choice does not work on this card -- see the fallback in
# start_server. Recorded with the results, because it changes the numbers.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-${VLLM_ATTENTION_BACKEND:-}}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
CONFIG="${CONFIG:-config/default.json}"

# Keep the 31 GB of weights on the persistent volume, not on the container
# disk: on RunPod the container disk is small and is wiped when the pod is
# rebuilt, and /workspace is the volume.
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

MIN_FREE_GB="${MIN_FREE_GB:-60}"                    # on the volume, for the model
MIN_CONTAINER_FREE_GB="${MIN_CONTAINER_FREE_GB:-25}"  # on the container disk, for vLLM
SERVER_START_TIMEOUT_S="${SERVER_START_TIMEOUT_S:-900}"
STATE_DIR="${STATE_DIR:-$WORKSPACE/.stresstest}"
SERVER_LOG="$STATE_DIR/vllm.log"
SERVER_PID_FILE="$STATE_DIR/vllm.pid"
DEADMAN_PID_FILE="$STATE_DIR/deadman.pid"
RUN_LOCK_FILE="$STATE_DIR/run.lock"

# Sticky across server starts: once we know this box needs vLLM's own sampler,
# every later engine variant starts with it straight away instead of burning a
# doomed start-up first. An operator who already exported the variable gets the
# same treatment without the failed attempt.
NO_FLASHINFER_SAMPLER=0
[ "${VLLM_USE_FLASHINFER_SAMPLER:-}" = 0 ] && NO_FLASHINFER_SAMPLER=1

PY="${PY:-python3}"
# Mock mode drives the harness's built-in fake vLLM instead of a real one, so
# the whole wrapper can be rehearsed on a laptop before any money is spent.
MOCK="${MOCK:-0}"
FORCE=0
SHUTDOWN_WHEN_DONE=0
DEADMAN_HOURS=""
SKIP_LESSON=0
SKIP_ENGINE=0
EXTRA_SETS=()      # as the harness CLI wants them: --set k=v
EXTRA_KV=()        # the same overrides as bare k=v, for the helpers below
LESSON_FLAGS=""

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

_c() { if [ -t 2 ]; then printf '\033[%sm%s\033[0m' "$1" "$2"; else printf '%s' "$2"; fi; }
say()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
head_() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$(_c '1' "$*")" >&2; }
warn() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$(_c '33' "LET OP: $*")" >&2; }
# Clearing the ERR trap first: a deliberate stop should print its own reason,
# not that plus a generic "afgebroken op regel N" from the trap.
die()  { trap - ERR; printf '[%s] %s\n' "$(date +%H:%M:%S)" "$(_c '31' "FOUT: $*")" >&2; exit 1; }

on_error() { die "afgebroken op regel $1. Server-log: $SERVER_LOG"; }
trap 'on_error $LINENO' ERR

# --------------------------------------------------------------------------
# Hardware detection -- the harness needs the same numbers the server runs on,
# otherwise the conversion from KV percentage to gigabytes is wrong and
# question 3 ("zou 72 GB ook volstaan?") gets the wrong answer.
# --------------------------------------------------------------------------

detect_gpu() {
  GPU_NAME="${GPU_NAME:-onbekend}"
  GPU_COUNT=1
  if command -v nvidia-smi >/dev/null 2>&1; then
    local line
    line="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1 || true)"
    if [ -n "$line" ]; then
      GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
      [ "${GPU_NAME}" = "onbekend" ] && GPU_NAME="$(echo "$line" | cut -d, -f1 | sed 's/^ *//;s/ *$//')"
      local mib per_card
      mib="$(echo "$line" | cut -d, -f2 | tr -d ' ')"
      per_card="$(awk -v m="$mib" 'BEGIN{printf "%.1f", m/1024}')"
      # With tensor parallelism the pool is the sum of the cards.
      VRAM_GB="${VRAM_GB:-$(awk -v p="$per_card" -v n="$TENSOR_PARALLEL" 'BEGIN{printf "%.1f", p*n}')}"

      # Which RTX PRO 6000 you got matters: the Max-Q variant carries the same
      # 96 GB but runs at half the power budget, so it is a different
      # measurement. The marketing name does not always say so; the default
      # power limit does. Record it, because "op welke kaart is dit gemeten"
      # is the first question anyone asks of these results.
      GPU_WATTS="$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9.' || true)"
      if [ -n "$GPU_WATTS" ]; then
        GPU_NAME="$GPU_NAME (${GPU_WATTS%.*}W)"
        if awk -v w="$GPU_WATTS" 'BEGIN{exit !(w < 400)}'; then
          warn "deze kaart staat op ${GPU_WATTS%.*}W. Dat wijst op een Max-Q-variant, die trager is afgeregeld dan de gewone uitvoering."
          warn "Meten kan prima, maar noteer het: de uitkomst geldt dan voor die variant, niet voor de kaart in de aanvraag."
        fi
      fi
    fi
  fi
  VRAM_GB="${VRAM_GB:-96.0}"
}

model_dir() { echo "$HF_HOME/hub/models--${MODEL//\//--}"; }

model_size_gb() {
  local dir; dir="$(model_dir)"
  [ -d "$dir" ] || { echo "0"; return; }
  du -sb --dereference "$dir" 2>/dev/null | awk '{printf "%.1f", $1/1024/1024/1024}' | grep . || echo 0
}

# Weights plus CUDA and activation overhead. The config ships 33.0 for a 31 GB
# model, i.e. about two gigabytes of slack; keep that ratio when measured.
weights_gb() {
  local size; size="$(model_size_gb)"
  awk -v s="$size" 'BEGIN{ if (s < 1) print ""; else printf "%.1f", s + 2.0 }'
}

harness_sets() {
  SETS=(-c "$CONFIG"
        --set "endpoint.model=$SERVED_NAME"
        --set "endpoint.base_url=http://127.0.0.1:$PORT/v1"
        --set "endpoint.metrics_url=http://127.0.0.1:$PORT/metrics"
        --set "hardware.gpu_memory_utilization=$GPU_UTIL"
        --set "hardware.vram_gb=$VRAM_GB")
  [ "${GPU_NAME:-onbekend}" != "onbekend" ] && SETS+=(--set "hardware.gpu_name=$GPU_NAME")
  local w; w="$(weights_gb)"
  [ -n "$w" ] && SETS+=(--set "hardware.model_weights_gb=$w")
  local backend; backend="$(cat "$STATE_DIR/current_backend" 2>/dev/null || true)"
  [ -n "$backend" ] && SETS+=(--set "hardware.attention_backend=$backend")
  local kv; kv="$(cat "$STATE_DIR/current_kv_dtype" 2>/dev/null || true)"
  [ -n "$kv" ] && SETS+=(--set "hardware.kv_cache_dtype=$kv")
  SETS+=("${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"}")
}

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

preflight() {
  head_ "Controle vooraf"
  cd "$REPO_DIR"
  mkdir -p "$STATE_DIR"

  if [ "$MOCK" = 1 ]; then
    GPU_NAME="${GPU_NAME:-nep-vLLM (simulatie)}"
    VRAM_GB="${VRAM_GB:-96.0}"
    say "MOCK: geen GPU, geen model -- alleen de nep-server. De getallen zeggen niets over echte hardware."
    return
  fi
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi ontbreekt -- dit is geen GPU-instance."
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >&2

  local driver major
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
  major="${driver%%.*}"
  if [ -n "$major" ] && [ "$major" -lt 570 ] 2>/dev/null; then
    warn "driver $driver is ouder dan 570. Blackwell (sm_120) heeft 570+ en CUDA 12.8+ nodig."
    warn "Kies een andere image; drivers bijwerken op een gehuurde instance is zelden de moeite waard."
  fi

  local free_gb
  free_gb="$(df -BG --output=avail "$WORKSPACE" 2>/dev/null | tail -1 | tr -dc '0-9')"
  if [ -n "$free_gb" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    [ "$FORCE" = 1 ] || die "nog $free_gb GB vrij op $WORKSPACE, minimaal $MIN_FREE_GB GB nodig (model is ~31 GB). Gebruik --force om toch door te gaan."
    warn "weinig schijfruimte ($free_gb GB), doorgaan op eigen risico"
  fi

  "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
    || die "Python 3.9 of nieuwer nodig, gevonden: $("$PY" --version 2>&1)"

  detect_gpu
  say "kaart: $GPU_NAME x$GPU_COUNT | pool voor het harnas: ${VRAM_GB} GB | HF_HOME=$HF_HOME"
}

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

ensure_vllm() {
  [ "$MOCK" = 1 ] && { say "MOCK: vLLM niet nodig"; return; }
  if command -v vllm >/dev/null 2>&1; then
    say "vLLM aanwezig: $(vllm --version 2>&1 | tail -1)"
    return
  fi
  head_ "vLLM installeren (dit duurt een paar minuten)"

  # This lands on the container disk, not on the volume the preflight checked.
  # vLLM brings its own pinned torch plus a stack of CUDA wheels, so it
  # replaces the torch that came with the image -- perfectly normal, but it
  # needs room. On RunPod the container disk is separate from /workspace and
  # is easy to leave at a default that is too small; running out halfway
  # through leaves a broken environment and an unclear error.
  local root_free
  root_free="$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
  if [ -n "$root_free" ] && [ "$root_free" -lt "$MIN_CONTAINER_FREE_GB" ]; then
    [ "$FORCE" = 1 ] || die "nog $root_free GB vrij op de container disk (/), minimaal $MIN_CONTAINER_FREE_GB GB nodig om vLLM en zijn CUDA-wheels te installeren. Op RunPod vergroot je dat bij Storage -> Container disk. Gebruik --force om toch door te gaan."
    warn "weinig ruimte op de container disk ($root_free GB), doorgaan op eigen risico"
  fi

  "$PY" -m pip install --upgrade pip >&2
  # --no-cache-dir: the wheel cache lands on the same container disk and would
  # roughly double the peak usage for gigabytes we never read again.
  "$PY" -m pip install --no-cache-dir "vllm>=0.10.0" >&2
  command -v vllm >/dev/null 2>&1 || die "vLLM is geinstalleerd maar staat niet op PATH."
  say "vLLM: $(vllm --version 2>&1 | tail -1)"
}

ensure_python_deps() {
  head_ "Harnasafhankelijkheden"
  # tokenizers is not optional in practice: without it the context sizes are
  # estimated, and context size is one of the two axes of the whole matrix.
  if "$PY" -c 'import tokenizers' 2>/dev/null; then
    say "tokenizers aanwezig -- contextgroottes worden exact geteld"
  else
    "$PY" -m pip install tokenizers >&2 \
      || die "tokenizers kon niet geinstalleerd worden. Zonder tokenizer zijn de contextgroottes een schatting; draai eerst 'stresstest calibrate' of los dit op."
  fi
  "$PY" -m pip install matplotlib >&2 || warn "matplotlib niet geinstalleerd; alleen SVG-grafieken (geen bezwaar)"
  [ "$MOCK" = 1 ] || "$PY" -m pip install "huggingface_hub[cli]" hf_transfer >&2
}

download_model() {
  [ "$MOCK" = 1 ] && { say "MOCK: model niet nodig"; return; }
  head_ "Model ophalen: $MODEL"
  local dir size
  dir="$(model_dir)"
  size="$(model_size_gb)"
  if awk -v s="$size" 'BEGIN{exit !(s > 25)}'; then
    say "staat er al: $dir (${size} GB), download overgeslagen"
    return
  fi
  [ "$size" != "0" ] && warn "onvolledige download gevonden (${size} GB), wordt hervat"

  mkdir -p "$HF_HOME"
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL" >&2
  else
    huggingface-cli download "$MODEL" >&2
  fi

  size="$(model_size_gb)"
  awk -v s="$size" 'BEGIN{exit !(s > 25)}' \
    || die "na de download staat er maar ${size} GB in $dir. Waarschijnlijk zijn alleen de metadata binnengehaald en is de download afgebroken. Draai dit commando opnieuw."
  say "binnen: ${size} GB in $dir"
}

fetch_corpus() {
  head_ "Corpus ophalen"
  "$PY" -m stresstest corpus -c "$CONFIG" >&2
}

cmd_setup() {
  preflight
  ensure_vllm
  ensure_python_deps
  download_model
  fetch_corpus
  head_ "Setup klaar"
}

# --------------------------------------------------------------------------
# The vLLM server
# --------------------------------------------------------------------------

server_running() {
  [ -f "$SERVER_PID_FILE" ] || return 1
  local pid; pid="$(cat "$SERVER_PID_FILE")"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# Two runs side by side share one GPU, one port and one results directory, and
# neither measurement means anything afterwards. The lock is held on an open
# file descriptor, so it dies with the process: a run that is killed, or a pod
# that is stopped mid-run, leaves nothing stale behind.
#
# Background children must not inherit it -- the deadman outlives the run by
# design -- so every spawn closes fd 9 with 9>&-.
running_run_pid() {
  local pid
  [ -f "$RUN_LOCK_FILE" ] || return 1
  pid="$(cat "$RUN_LOCK_FILE" 2>/dev/null || true)"
  [ -n "$pid" ] && [ "$pid" != "$$" ] && kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

die_second_run() {
  die "er draait al een run (pid $1). Twee runs tegelijk delen dezelfde GPU, dezelfde
      poort en dezelfde resultaatmap; wat daaruit komt meet niets.
      meekijken:  scripts/pod.sh log -f
      overnemen:  scripts/pod.sh stop    (stopt de run en vLLM), daarna opnieuw"
}

claim_run() {
  mkdir -p "$STATE_DIR"
  exec 9>>"$RUN_LOCK_FILE" || die "kan $RUN_LOCK_FILE niet openen"
  local other=""
  if command -v flock >/dev/null 2>&1; then
    flock -n 9 || other="$(cat "$RUN_LOCK_FILE" 2>/dev/null || true)"
  else
    # No flock on this image: fall back to the pid in the file. Weaker (two
    # starts in the same second can both win) but better than nothing.
    other="$(running_run_pid || true)"
  fi
  [ -z "$other" ] || die_second_run "$other"
  printf '%s' "$$" > "$RUN_LOCK_FILE"
  # Truncate rather than remove: another start may already have the same inode
  # open, and deleting the path would let it hand out a second lock.
  trap 'printf "" > "$RUN_LOCK_FILE" 2>/dev/null || true' EXIT
}

stop_server() {
  if server_running; then
    local pid; pid="$(cat "$SERVER_PID_FILE")"
    say "vLLM stoppen (pid $pid)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
  pkill -f "vllm serve" 2>/dev/null || true
  rm -f "$SERVER_PID_FILE"
  # Give the driver a moment to hand the memory back before the next start.
  sleep 5
}

wait_ready() {
  local deadline=$((SECONDS + SERVER_START_TIMEOUT_S)) pid
  pid="$(cat "$SERVER_PID_FILE")"
  while [ $SECONDS -lt $deadline ]; do
    if ! kill -0 "$pid" 2>/dev/null; then return 2; fi
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}

# vLLM prints a Python traceback on failure, and its last lines are the least
# informative part: the real reason sits further up. Surface that first, then
# the tail, so the operator does not have to go spelunking in a 500-line log.
SERVER_ERROR_PATTERNS='no available memory|out of memory|CUDA out of memory|compute capability|not supported|unrecognized arguments|invalid choice|does not exist|ValueError|RuntimeError|Error'

show_server_error() {
  local log="$1" hits
  [ -f "$log" ] || return 0
  hits="$(grep -nEi "$SERVER_ERROR_PATTERNS" "$log" 2>/dev/null | grep -vE '^\s*[0-9]+:\s*File "' | tail -12 || true)"
  if [ -n "$hits" ]; then
    printf '%s\n' "--- vermoedelijke oorzaak, uit $log ---" >&2
    printf '%s\n' "$hits" >&2
  fi
  printf '%s\n' "--- laatste 25 regels van $log ---" >&2
  tail -n 25 "$log" >&2 || true
}

# start_server [tunable flags]. Without an argument the baseline is used.
# start_server <tunable flags> [strict]
# In strict mode nothing is silently substituted and a failure returns 1
# instead of ending the run: the caller decides what to do.
start_server() {
  local tunable="${1:-}"
  local strict=0
  [ "${2:-}" = strict ] && strict=1
  [ -n "$tunable" ] || tunable="--kv-cache-dtype $KV_CACHE_DTYPE --max-num-seqs $MAX_NUM_SEQS --max-model-len $MAX_MODEL_LEN"

  stop_server

  if [ "$MOCK" = 1 ]; then
    # The fake server takes the same two tunables under the same names; it has
    # no notion of a KV dtype, so that one is dropped.
    local mock_flags
    mock_flags="$(echo "$tunable" | sed 's/--kv-cache-dtype [a-z0-9]*//')"
    local cmd="$PY -m stresstest mock -- --host 127.0.0.1 --port $PORT --gpu-memory-gb $VRAM_GB $mock_flags"
    say "start: $cmd"
    : > "$SERVER_LOG"
    nohup bash -c "$cmd" >>"$SERVER_LOG" 2>&1 9>&- &
    echo $! > "$SERVER_PID_FILE"
    wait_ready || { tail -20 "$SERVER_LOG" >&2 || true; die "nep-server kwam niet omhoog"; }
    say "nep-server draait (poort $PORT)"
    echo "$tunable" > "$STATE_DIR/current_flags"
    return 0
  fi

  local tool_flags="--enable-auto-tool-choice --tool-call-parser qwen3_coder"
  local backend="$ATTENTION_BACKEND"
  local server_env attempt
  for attempt in 1 2 3; do
    local cmd="vllm serve $MODEL --served-model-name $SERVED_NAME --host $HOST_BIND --port $PORT"
    cmd="$cmd --gpu-memory-utilization $GPU_UTIL --enable-prefix-caching $tunable"
    [ "$TENSOR_PARALLEL" -gt 1 ] && cmd="$cmd --tensor-parallel-size $TENSOR_PARALLEL"
    cmd="$cmd $tool_flags"

    say "start: $cmd"
    server_env=(HF_HOME="$HF_HOME")
    if [ -n "$backend" ]; then
      server_env+=(VLLM_ATTENTION_BACKEND="$backend")
      say "       VLLM_ATTENTION_BACKEND=$backend"
    fi
    if [ "$NO_FLASHINFER_SAMPLER" = 1 ]; then
      server_env+=(VLLM_USE_FLASHINFER_SAMPLER=0)
      say "       VLLM_USE_FLASHINFER_SAMPLER=0"
    fi
    [ -f "$SERVER_LOG" ] && mv -f "$SERVER_LOG" "$SERVER_LOG.vorige"
    : > "$SERVER_LOG"
    nohup env "${server_env[@]}" bash -c "$cmd" >>"$SERVER_LOG" 2>&1 9>&- &
    echo $! > "$SERVER_PID_FILE"

    local status=0
    wait_ready || status=$?
    if [ "$status" = 0 ]; then
      say "vLLM draait (poort $PORT). Log: $SERVER_LOG"
      echo "$tunable" > "$STATE_DIR/current_flags"
      echo "$backend" > "$STATE_DIR/current_backend"
      printf '%s' "$tunable" | sed -n 's/.*--kv-cache-dtype \([a-z0-9]*\).*/\1/p' \
        > "$STATE_DIR/current_kv_dtype"
      return 0
    fi

    # FlashInfer is used for top-k/top-p sampling regardless of the attention
    # backend, and its JIT compiler refuses to build here: the log says
    # "Failed to get device capability: SM 12.x requires CUDA >= 12.9", after
    # which check_cuda_arch() reports the misleading "FlashInfer requires GPUs
    # with sm75 or higher" -- the card is sm_120. The toolkit on the image is
    # simply older than what this flashinfer needs for Blackwell.
    #
    # vLLM can sample without it, so try that. This changes nothing about what
    # is measured -- the harness runs at temperature 0 -- so it is safe to do
    # for the engine variants too.
    if [ "$status" = 2 ] && [ "$NO_FLASHINFER_SAMPLER" != 1 ] \
       && grep -qEi "FlashInfer requires GPUs|check_cuda_arch|SM 12\.x requires CUDA|No module named .flashinfer" \
                    "$SERVER_LOG" 2>/dev/null; then
      warn "vLLM struikelt over FlashInfer; opnieuw zonder de FlashInfer-sampler."
      warn "Dat raakt de meting niet: het harnas draait op temperatuur 0."
      NO_FLASHINFER_SAMPLER=1
      continue
    fi

    # The README's documented fallback, but only when the log actually blames
    # the tool-call parser. Retrying blindly costs another start-up and, worse,
    # points the operator at the wrong thing: an engine that dies on memory or
    # on missing kernels dies again in exactly the same way.
    if [ "$status" = 2 ] && [ -n "$tool_flags" ] \
       && grep -qEi 'tool.call.parser|enable-auto-tool-choice|unrecognized arguments|invalid choice' \
                    "$SERVER_LOG" 2>/dev/null; then
      warn "de tool-call-parser wordt niet geaccepteerd; opnieuw zonder die twee vlaggen"
      warn "(het harnas schakelt dan zelf over op een tekstvariant met dezelfde berichtstructuur)"
      tool_flags=""
      continue
    fi

    show_server_error "$SERVER_LOG"
    if [ "$strict" = 1 ]; then
      return 1
    fi
    if [ "$status" = 2 ]; then
      if grep -q "No module named .flashinfer" "$SERVER_LOG" 2>/dev/null; then
        die "vLLM importeert flashinfer ook als hij het niet gebruikt, en het pakket is hier weg. Zet het terug ( pip install flashinfer-python ) en start opnieuw; dit script zet zelf VLLM_USE_FLASHINFER_SAMPLER=0 zodat de JIT-compiler er niet aan te pas komt."
      fi
      if grep -qEi 'FlashInfer requires GPUs|check_cuda_arch' "$SERVER_LOG" 2>/dev/null; then
        die "vLLM blijft op FlashInfer stuklopen, ook zonder de FlashInfer-sampler. De JIT-compiler van FlashInfer kan deze kaart (sm_120) niet bouwen met de CUDA-toolkit in deze image; het log noemt CUDA >= 12.9. Een image met een nieuwere toolkit is dan de uitweg -- /workspace blijft staan, dus het model hoeft niet opnieuw gedownload."
      fi
      die "vLLM is tijdens het opstarten gestopt. Zie hierboven en $SERVER_LOG. Vaakst voorkomend: te weinig geheugen voor de KV-cache (verlaag --max-model-len of GPU_UTIL), of een vLLM zonder kernels voor deze kaart."
    fi
    die "vLLM kwam niet omhoog binnen ${SERVER_START_TIMEOUT_S}s, maar draait nog wel. Zie $SERVER_LOG; verhoog zo nodig SERVER_START_TIMEOUT_S."
  done
}

# Counting series is too crude: what matters is whether the three numbers the
# main question hangs on are actually exposed. Without preemptions, prefix
# cache hits and KV occupancy you are only measuring latency, which is half
# the question.
check_metrics() {
  local body count missing=""
  body="$(curl -s "http://127.0.0.1:$PORT/metrics" || true)"
  count="$(printf '%s' "$body" | grep -c '^vllm:' || true)"

  printf '%s' "$body" | grep -q '^vllm:\(num_preemptions\)' || missing="$missing preempties"
  printf '%s' "$body" | grep -q '^vllm:\(gpu_\)\?prefix_cache_\(queries\|hits\|hit_rate\)' || missing="$missing prefix-cache"
  printf '%s' "$body" | grep -q '^vllm:\(kv_cache_usage_perc\|gpu_cache_usage_perc\)' || missing="$missing KV-bezetting"

  if [ -n "$missing" ]; then
    [ "$FORCE" = 1 ] || die "/metrics mist:$missing (van $count vllm-reeksen). Daarmee is de hoofdvraag niet te beantwoorden -- zonder preempties, cache hit rate en KV-bezetting meet je alleen latentie. Zet de Prometheus-endpoint aan, of gebruik --force."
    warn "/metrics mist:$missing -- doorgaan vanwege --force; de conclusie wordt onvolledig"
  else
    say "/metrics: $count reeksen, met preempties, prefix-cache en KV-bezetting"
  fi
}

cmd_serve() { preflight; start_server "${1:-}"; check_metrics; }
cmd_stop() {
  local pid=""
  pid="$(running_run_pid || true)"
  if [ -n "$pid" ]; then
    say "lopende run stoppen (pid $pid)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
  stop_server
  say "gestopt"
}

cmd_status() {
  local run_pid=""
  run_pid="$(running_run_pid || true)"
  if [ -n "$run_pid" ]; then
    say "er draait een run (pid $run_pid) -- meekijken: scripts/pod.sh log -f"
  else
    say "er draait geen run"
  fi
  if server_running; then
    say "vLLM draait (pid $(cat "$SERVER_PID_FILE")), vlaggen: $(cat "$STATE_DIR/current_flags" 2>/dev/null || echo onbekend)"
    curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 400 >&2 || true; echo >&2
  else
    say "vLLM draait niet"
  fi
  if [ -f "$DEADMAN_PID_FILE" ] && kill -0 "$(cat "$DEADMAN_PID_FILE")" 2>/dev/null; then
    say "doodsklok actief (pid $(cat "$DEADMAN_PID_FILE")) -- $(cat "$STATE_DIR/deadman_at" 2>/dev/null || echo '?')"
  fi
}

# --------------------------------------------------------------------------
# Driving the harness
# --------------------------------------------------------------------------

# Run ids in a group that do not yet have a run.json on disk. This is what
# makes the whole thing resumable after a dropped connection.
pending_ids() {
  local group="$1" dir="$2"; shift 2
  "$PY" - "$group" "$dir" "$CONFIG" "$@" <<'PY'
import os, sys
from stresstest.cli import load_config
from stresstest.matrix import BUILDERS
group, directory, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
config = load_config(config_path, sys.argv[4:])
ids = [spec.run_id for spec in BUILDERS[group](config)]
print(" ".join(i for i in ids
               if not os.path.exists(os.path.join(directory, "runs", i, "run.json"))))
PY
}

engine_variants() {
  "$PY" - "$CONFIG" "${EXTRA_KV[@]+"${EXTRA_KV[@]}"}" <<'PY'
import sys
from stresstest.cli import load_config
block = load_config(sys.argv[1], sys.argv[2:]).get("matrix", {}).get("engine_variation", {})
if block.get("enabled", True):
    context = int(block.get("context_tokens", 32000))
    for variant in block.get("variants", []):
        print("\t".join([variant["name"], variant.get("flags", ""), str(context)]))
PY
}

plan_duration() {
  "$PY" - "$CONFIG" "$@" <<'PY'
import sys
from stresstest.cli import load_config
from stresstest.matrix import build_specs, estimate_duration_s, PHASE1, BUILDERS
config = load_config(sys.argv[1], sys.argv[2:])
gap = float(config.get("run_defaults", {}).get("gap_between_runs_s", 30))
phase1 = estimate_duration_s(build_specs(config, PHASE1), gap)
lesson = estimate_duration_s(BUILDERS["lesson"](config), gap)
print(int(phase1 + lesson))
PY
}

# run_group <group> <results dir>
run_group() {
  local group="$1" dir="$2"
  local pending
  pending="$(pending_ids "$group" "$dir" "${EXTRA_KV[@]+"${EXTRA_KV[@]}"}")"
  if [ -z "$pending" ]; then
    say "groep $group: alles staat al in $dir, overslaan"
    return 0
  fi
  head_ "Groep $group ($(echo "$pending" | wc -w | tr -d ' ') runs)"
  harness_sets
  # shellcheck disable=SC2086 -- word splitting of the id list is intended
  "$PY" -m stresstest matrix --only "$group" --out "$dir" --run $pending --no-pause "${SETS[@]}"
}

# The engine variants cannot be driven from the client: each one needs its own
# vLLM. The wrapper restarts the server itself, which is why --no-pause is
# correct here and meaningless without it.
run_engine_group() {
  local dir="$1" name flags context need line
  head_ "Engine-varianten (elke variant herstart vLLM)"
  while IFS=$'\t' read -r name flags context; do
    [ -n "$name" ] || continue
    if [ -f "$dir/runs/engine_$name/run.json" ]; then
      say "engine_$name staat al in $dir, overslaan"
      continue
    fi
    # A run of N context tokens sends up to 1024 tokens of headroom on top; if
    # the variant's window is smaller than that, vLLM rejects every request.
    local window; window="$(echo "$flags" | sed -n 's/.*--max-model-len \([0-9]*\).*/\1/p')"
    need=$((context + 1024))
    if [ -n "$window" ] && [ "$window" -lt "$need" ]; then
      warn "variant $name heeft --max-model-len $window maar de run vraagt ~$need tokens; overslaan"
      continue
    fi
    say "variant $name: $flags"
    if ! start_server "$flags" strict; then
      warn "variant $name start niet op deze opstelling; overgeslagen (zie $SERVER_LOG)"
      warn "de overige varianten gaan gewoon door"
      continue
    fi
    check_metrics
    harness_sets
    "$PY" -m stresstest matrix --run "engine_$name" --only engine --out "$dir" --no-pause "${SETS[@]}"
  done < <(engine_variants)
}

# Newest run log, or empty. Reading the log is the thing an operator does most
# often, and `tail -20 .../pod-*.log` stops working the moment a second run
# leaves a second log behind: with more than one file operand the obsolete
# -NUM form is rejected ("option used in invalid context"). Hence a command
# that always names exactly one file.
latest_log() {
  ls -t "$STATE_DIR"/pod-*.log 2>/dev/null | head -1 || true
}

cmd_log() {
  local file
  file="$(latest_log)"
  [ -n "$file" ] || die "nog geen logbestand in $STATE_DIR -- draai eerst 'scripts/pod.sh all --detach'."
  if [ "${1:-}" = "-f" ]; then
    say "volgen: $file  (ctrl-C stopt alleen het meekijken, niet de test)"
    tail -n 40 -f "$file"
  else
    say "$file"
    tail -n "${1:-40}" "$file"
  fi
}

cmd_doctor() { harness_sets; "$PY" -m stresstest doctor "${SETS[@]}"; }
cmd_plan()   { harness_sets; "$PY" -m stresstest plan "${SETS[@]}" "$@"; }

cmd_report() {
  local dir="${1:-}"
  [ -n "$dir" ] || dir="$(latest_matrix_dir)"
  [ -n "$dir" ] || die "geen resultatenmap gevonden; geef er een mee: pod.sh report results/<map>"
  "$PY" -m stresstest report "$dir"
}

cmd_lesson() {
  preflight
  local dir="results/$(date +%Y%m%d-%H%M%S)_les"
  if [ -n "$LESSON_FLAGS" ]; then
    say "lesvalidatie op afwijkende serverinstelling: $LESSON_FLAGS"
    start_server "$LESSON_FLAGS"
  elif ! server_running; then
    start_server
  fi
  check_metrics
  harness_sets
  "$PY" -m stresstest lesson --out "$dir" "${SETS[@]}"
  echo "$dir" > "$STATE_DIR/last_lesson_dir"
}

# --------------------------------------------------------------------------
# Not forgetting the instance -- the most expensive mistake in this test
# --------------------------------------------------------------------------

# Stops the pod rather than terminating it: GPU billing stops, /workspace and
# the results survive, and you terminate by hand once you have copied them.
power_down() {
  [ "$MOCK" = 1 ] && { warn "MOCK: de pod zou nu gestopt worden; dat gebeurt niet in mock-modus."; stop_server; return 0; }
  say "pod stoppen"
  stop_server
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID" && return 0
    warn "runpodctl stop is mislukt"
  fi
  warn "geen runpodctl; de machine wordt uitgezet. Controleer in het dashboard of het verbruik echt stopt."
  poweroff || shutdown -h now || true
}

arm_deadman() {
  local hours="$1"
  disarm_deadman
  local seconds; seconds="$(awk -v h="$hours" 'BEGIN{printf "%d", h*3600}')"
  date -d "+${seconds} seconds" '+%Y-%m-%d %H:%M:%S' > "$STATE_DIR/deadman_at" 2>/dev/null || echo "over ${hours}u" > "$STATE_DIR/deadman_at"
  # MOCK travels with it: a rehearsal on a laptop must never power off a laptop.
  nohup bash -c "sleep $seconds; MOCK=$MOCK WORKSPACE='$WORKSPACE' '$REPO_DIR/scripts/pod.sh' power-down" >>"$STATE_DIR/deadman.log" 2>&1 9>&- &
  echo $! > "$DEADMAN_PID_FILE"
  warn "doodsklok gezet: over ${hours} uur ($(cat "$STATE_DIR/deadman_at")) wordt de pod gestopt."
  warn "Afzetten met: scripts/pod.sh disarm"
}

disarm_deadman() {
  if [ -f "$DEADMAN_PID_FILE" ]; then
    kill "$(cat "$DEADMAN_PID_FILE")" 2>/dev/null || true
    rm -f "$DEADMAN_PID_FILE" "$STATE_DIR/deadman_at"
    say "doodsklok afgezet"
  fi
}

archive_results() {
  local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
  local tarball="$WORKSPACE/stresstest-results-$stamp.tar.gz"
  tar -czf "$tarball" -C "$REPO_DIR" results 2>/dev/null || return 0
  say "resultaten ingepakt: $tarball ($(du -h "$tarball" | cut -f1))"
  echo "$tarball"
}

# --------------------------------------------------------------------------
# The whole thing
# --------------------------------------------------------------------------

# Most recent phase-1 results directory, or empty if there is none yet.
# `ls` fails when results/ does not exist, and under `set -o pipefail` that
# failure escapes the command substitution and aborts the run -- which is
# exactly what happens on the very first, clean start.
latest_matrix_dir() {
  ls -dt results/*_matrix 2>/dev/null | head -1 || true
}

cmd_all() {
  local started=$SECONDS
  cmd_setup

  local dir="${RESULTS_DIR:-}"
  if [ -z "$dir" ]; then
    dir="$(latest_matrix_dir)"
    if [ -n "$dir" ]; then
      say "hervat in bestaande map $dir (zet RESULTS_DIR= om ergens anders te beginnen)"
    else
      dir="results/$(date +%Y%m%d-%H%M%S)_matrix"
    fi
  fi
  mkdir -p "$dir"

  if [ -n "$DEADMAN_HOURS" ]; then
    if [ "$DEADMAN_HOURS" = auto ]; then
      local estimate; estimate="$(plan_duration "${EXTRA_KV[@]+"${EXTRA_KV[@]}"}")"
      DEADMAN_HOURS="$(awk -v s="$estimate" 'BEGIN{printf "%.1f", s/3600 + 1.5}')"
    fi
    arm_deadman "$DEADMAN_HOURS"
  fi

  start_server
  check_metrics

  head_ "Doctor"
  if ! cmd_doctor; then
    [ "$FORCE" = 1 ] || die "doctor meldt problemen -- los die eerst op. Een uur meten met een kapotte opstelling is een verloren uur. Gebruik --force om toch door te gaan."
    warn "doctor meldt problemen, doorgaan vanwege --force"
  fi

  # rampup first: it is the cheapest useful number and it doubles as a canary
  # for the whole chain. Then the rest, cheapest-insight-first.
  run_group rampup "$dir"
  run_group sweep "$dir"
  run_group scenarios "$dir"
  run_group shared "$dir"
  run_group activity "$dir"
  [ "$SKIP_ENGINE" = 1 ] || run_engine_group "$dir"

  head_ "Fase 1 afronden"
  start_server            # back to the baseline configuration
  "$PY" -m stresstest report "$dir"

  local lesson_dir=""
  if [ "$SKIP_LESSON" = 0 ]; then
    head_ "Fase 2: lesvalidatie van negentig minuten"
    cmd_lesson
    lesson_dir="$(cat "$STATE_DIR/last_lesson_dir" 2>/dev/null || true)"
  fi

  head_ "Klaar in $(awk -v s=$((SECONDS - started)) 'BEGIN{printf "%du%02dm", s/3600, (s%3600)/60}')"
  say "fase 1: $dir/RESULTATEN.md"
  [ -n "$lesson_dir" ] && say "fase 2: $lesson_dir/RESULTATEN.md"
  archive_results >/dev/null || true
  echo >&2
  say "Haal de resultaten op, vanaf je eigen machine:"
  say "    scp -r root@<ip>:$REPO_DIR/results ./results-van-de-gpu"
  echo >&2
  warn "ZET DE INSTANCE UIT als je klaar bent. Terminate, niet Stop -- bij Stop tikt de opslag door."

  if [ "$SHUTDOWN_WHEN_DONE" = 1 ]; then
    warn "de pod wordt over 10 minuten gestopt (--shutdown). Afbreken: scripts/pod.sh disarm"
    arm_deadman 0.17
  else
    disarm_deadman
  fi
}

# --------------------------------------------------------------------------

usage() {
  cat >&2 <<'USAGE'
scripts/pod.sh <commando> [opties]

Commando's
  all                de hele test onbewaakt: setup, fase 1, engine-varianten,
                     fase 2, conclusie. Hervat wat al gedraaid is.
  setup              installeren, model ophalen, corpus ophalen
  serve [vlaggen]    vLLM starten (standaard de basisinstelling)
  stop               vLLM stoppen
  status             wat draait er, en staat de doodsklok aan
  log [-f|<n>]       de laatste regels van het nieuwste logbestand; -f volgt mee
  doctor             controle van endpoint, metrics, corpus, contextvenster
  plan               de runs en de geschatte huurkosten
  group <naam>       een losse groep: rampup, sweep, scenarios, shared,
                     activity, engine
  lesson             alleen fase 2
  report [map]       grafieken en conclusie opnieuw maken (geen GPU nodig)
  deadman <uren>     doodsklok zetten: stop de pod automatisch
  disarm             doodsklok afzetten
  power-down         nu stoppen

Opties
  --force            doorgaan ondanks waarschuwingen van doctor of /metrics
  --mock             generale repetitie op je laptop tegen de ingebouwde
                     nep-vLLM: geen GPU, geen model, geen kosten
  --deadman <uren>   doodsklok zetten bij 'all'; 'auto' = geschatte duur + 1,5u
  --shutdown         de pod stoppen zodra alles klaar is
  --skip-lesson      fase 2 overslaan
  --skip-engine      de engine-varianten overslaan (die herstarten vLLM)
  --flags "<vlaggen>"  afwijkende vLLM-vlaggen voor 'lesson'
  -c, --config <pad> ander configuratiebestand (standaard config/default.json)
  --set k=v          configuratie overschrijven, wordt doorgegeven aan het harnas
  --detach           zichzelf loskoppelen van de terminal (nohup) en het log tonen

Omgevingsvariabelen
  MODEL SERVED_NAME PORT MAX_MODEL_LEN MAX_NUM_SEQS KV_CACHE_DTYPE GPU_UTIL
  MIN_FREE_GB MIN_CONTAINER_FREE_GB ATTENTION_BACKEND
  TENSOR_PARALLEL VRAM_GB GPU_NAME HF_HOME CONFIG RESULTS_DIR WORKSPACE

Voorbeelden
  scripts/pod.sh all --deadman auto
  TENSOR_PARALLEL=2 VRAM_GB=64 scripts/pod.sh all      # twee RTX 5090's
  MODEL=Qwen/Qwen2.5-Coder-7B-Instruct MAX_MODEL_LEN=32768 \
    VRAM_GB=24 scripts/pod.sh all --skip-engine         # goedkoop uitproberen
USAGE
}

main() {
  # Kept verbatim so --detach can hand the exact same invocation to nohup.
  local raw=("$@")
  local command="${1:-}"; shift || true
  local positional=()
  local detach=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --force) FORCE=1; shift ;;
      --mock) MOCK=1; SERVED_NAME="${SERVED_NAME_OVERRIDE:-mock-model}"; shift ;;
      --shutdown) SHUTDOWN_WHEN_DONE=1; shift ;;
      --skip-lesson) SKIP_LESSON=1; shift ;;
      --skip-engine) SKIP_ENGINE=1; shift ;;
      --deadman) DEADMAN_HOURS="$2"; shift 2 ;;
      --flags) LESSON_FLAGS="$2"; shift 2 ;;
      --set) EXTRA_SETS+=(--set "$2"); EXTRA_KV+=("$2"); shift 2 ;;
      --detach) detach=1; shift ;;
      -c|--config) CONFIG="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      --) shift; while [ $# -gt 0 ]; do positional+=("$1"); shift; done ;;
      -*) case "$command" in plan|log) positional+=("$1"); shift; continue ;; esac
          usage; die "onbekende optie: $1" ;;
      *) positional+=("$1"); shift ;;
    esac
  done
  set -- "${positional[@]+"${positional[@]}"}"

  cd "$REPO_DIR"

  if [ "$detach" = 1 ]; then
    local busy; busy="$(running_run_pid || true)"
    [ -z "$busy" ] || die_second_run "$busy"
    mkdir -p "$STATE_DIR"
    local log="$STATE_DIR/pod-$(date +%Y%m%d-%H%M%S).log"
    local forwarded=()
    local argument
    for argument in "${raw[@]}"; do
      [ "$argument" = "--detach" ] || forwarded+=("$argument")
    done
    say "losgekoppeld; volgen met:  tail -f $log"
    STRESSTEST_DETACHED=1 nohup "$0" "${forwarded[@]}" >"$log" 2>&1 9>&- &
    echo $! > "$STATE_DIR/pod.pid"
    say "pid $(cat "$STATE_DIR/pod.pid"); afbreken met: kill $(cat "$STATE_DIR/pod.pid")"
    exit 0
  fi

  if [ "$command" = all ] && [ -t 1 ] && [ -z "${TMUX:-}" ] && [ -z "${STRESSTEST_DETACHED:-}" ]; then
    warn "je draait niet in tmux. Valt de SSH-verbinding weg, dan stopt de test (de pod niet)."
    warn "Beter:  tmux new -s test   of   scripts/pod.sh all --detach"
    sleep 5
  fi

  case "$command" in
    all|serve|group|lesson) claim_run ;;
  esac

  case "$command" in
    all)        cmd_all ;;
    setup)      cmd_setup ;;
    serve)      cmd_serve "${1:-}" ;;
    stop)       cmd_stop ;;
    status)     cmd_status ;;
    log)        cmd_log "${1:-}" ;;
    doctor)     preflight; cmd_doctor ;;
    plan)       detect_gpu; cmd_plan "$@" ;;
    group)      [ $# -ge 1 ] || die "welke groep? rampup, sweep, scenarios, shared, activity, engine"
                preflight
                server_running || start_server
                local dir="${RESULTS_DIR:-$(latest_matrix_dir)}"
                [ -n "$dir" ] || dir="results/$(date +%Y%m%d-%H%M%S)_matrix"
                mkdir -p "$dir"
                if [ "$1" = engine ]; then run_engine_group "$dir"; else run_group "$1" "$dir"; fi
                say "resultaten in $dir (conclusie bijwerken: scripts/pod.sh report $dir)" ;;
    lesson)     cmd_lesson >/dev/null ;;
    report)     cmd_report "${1:-}" ;;
    deadman)    [ $# -ge 1 ] || die "hoeveel uur?"; mkdir -p "$STATE_DIR"; arm_deadman "$1" ;;
    disarm)     disarm_deadman ;;
    power-down) power_down ;;
    ""|-h|--help) usage ;;
    *) usage; die "onbekend commando: $command" ;;
  esac
}

main "$@"
