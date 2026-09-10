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
# Which attention backend vLLM uses, passed as --attention-backend. Empty means
# "let vLLM choose", which is right until its choice does not build on this
# card -- see the FlashInfer fallback in start_server, which then sets this to
# TRITON_ATTN for the rest of the run. Recorded with the results, because it
# changes the numbers. (VLLM_ATTENTION_BACKEND is accepted for operators who
# still export it; vLLM 0.28 itself no longer reads that variable.)
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

# Same idea for the Hugging Face Hub. vLLM asks the Hub for the repo's file
# list on every start, also when all 31 GB are already on disk, and the Hub
# rate-limits anonymous API calls per IP -- an IP a rented pod shares with
# every other container on the machine. Once we have served from the local
# snapshot, every later engine variant does the same straight away instead of
# burning a doomed start-up first.
HF_OFFLINE=0
[ "${HF_HUB_OFFLINE:-}" = 1 ] && HF_OFFLINE=1

# And again for the link between the cards. With --tensor-parallel-size above
# 1 the workers exchange activations over NCCL, which prefers the direct
# card-to-card path (PCIe peer-to-peer). Consumer Blackwell does not always
# have it -- two 5090s in a rented pod is exactly the case -- and NCCL then
# dies in the worker instead of routing around it. Sticky, like the rest: once
# we know, every later engine variant starts that way.
NCCL_P2P_OFF=0
[ "${NCCL_P2P_DISABLE:-}" = 1 ] && NCCL_P2P_OFF=1
# How long to wait out a 429 when there is no complete download to fall back
# on. Doubles per attempt; the Hub's anonymous window resets in minutes.
HF_RATE_LIMIT_WAIT_S="${HF_RATE_LIMIT_WAIT_S:-60}"
HF_RATE_LIMIT_MAX_WAIT_S="${HF_RATE_LIMIT_MAX_WAIT_S:-240}"
HF_DOWNLOAD_RETRIES="${HF_DOWNLOAD_RETRIES:-4}"
# How many times start_server may try. Six, not one: the two FlashInfer
# fallbacks (sampler, then attention backend) and the tool-parser fallback each
# cost an attempt, and waiting out a rate limit costs the rest.
SERVER_START_ATTEMPTS="${SERVER_START_ATTEMPTS:-6}"

PY="${PY:-python3}"
# Mock mode drives the harness's built-in fake vLLM instead of a real one, so
# the whole wrapper can be rehearsed on a laptop before any money is spent.
MOCK="${MOCK:-0}"
FORCE=0
SHUTDOWN_WHEN_DONE=0
DEADMAN_HOURS=""
SKIP_LESSON=0
SKIP_ENGINE=0
# Resume in a directory that was measured with a different corpus, behaviour
# model or tokenizer. Off by default: skipping runs that only share a name with
# what is wanted produces a complete-looking report over two measurements.
RESUME_ANYWAY=0
# Set by cmd_all when it actually wrote the combined RESULTATEN.md in the repo
# root this run. Without it that file is whatever a previous run left behind,
# and pushing it along suggests a coverage this measurement does not have.
REPORT_WRITTEN=0
# Push the results to the repository when 'all' finishes, and stop the pod once
# that push succeeded. Results that only exist on a rented machine are one
# forgotten terminate away from being gone, so this is on by default.
PUSH_RESULTS=1
RESULTS_BRANCH="${RESULTS_BRANCH:-}"
PUSH_RETRIES="${PUSH_RETRIES:-4}"
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
      # With tensor parallelism the pool is the sum of the cards -- of the
      # cards that are there. TENSOR_PARALLEL is what was asked for, and
      # asking for more cards than the machine has (see check_tensor_parallel,
      # which refuses it unless --force) would otherwise report a pool twice
      # the size of the machine and put every KV-gigabyte in the report out by
      # the same factor.
      local cards="$TENSOR_PARALLEL"
      [ "$cards" -gt "$GPU_COUNT" ] && cards="$GPU_COUNT"
      VRAM_GB="${VRAM_GB:-$(awk -v p="$per_card" -v n="$cards" 'BEGIN{printf "%.1f", p*n}')}"

      # Which RTX PRO 6000 you got matters: the Max-Q variant carries the same
      # 96 GB but runs at half the power budget, so it is a different
      # measurement. The marketing name does not always say so; the default
      # power limit does. Record it, because "op welke kaart is dit gemeten"
      # is the first question anyone asks of these results.
      GPU_WATTS="$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9.' || true)"
      if [ -n "$GPU_WATTS" ]; then
        # GPU_NAME survives between invocations (see the persisted-state list),
        # so appending unconditionally gives you "... (600W) (600W)" on the
        # second phase -- and two results directories that disagree about which
        # card they were measured on.
        case "$GPU_NAME" in
          *"(${GPU_WATTS%.*}W)") : ;;
          *) GPU_NAME="$GPU_NAME (${GPU_WATTS%.*}W)" ;;
        esac
        if awk -v w="$GPU_WATTS" 'BEGIN{exit !(w < 400)}'; then
          warn "deze kaart staat op ${GPU_WATTS%.*}W. Dat wijst op een Max-Q-variant, die trager is afgeregeld dan de gewone uitvoering."
          warn "Meten kan prima, maar noteer het: de uitkomst geldt dan voor die variant, niet voor de kaart in de aanvraag."
        fi
      fi
    fi
  fi
  VRAM_GB="${VRAM_GB:-96.0}"
}

# torch and the driver have to agree about CUDA, and an image does not
# guarantee that they do. A torch built against CUDA 13 needs a 580 driver; a
# pod with 570.144 offers CUDA 12.8, and every worker then dies in
# torch._C._cuda_init() with "The NVIDIA driver on your system is too old
# (found version 12080)". vLLM reports that as a background process that went
# away -- fifteen minutes in, with the model loaded, wrapped in a traceback
# about worker start-up that names neither torch nor the driver.
#
# Asking torch the same question costs a second and answers it in the
# operator's words. Skipped in silence when torch is not there yet: on a clean
# pod this runs before vLLM is installed, and cmd_setup asks again afterwards.
check_driver_matches_torch() {
  [ "$MOCK" = 1 ] && return 0
  local probe status=0
  probe="$("$PY" -c 'import sys
try:
    import torch
except Exception:
    sys.exit(3)
built = torch.version.cuda or "onbekend"
try:
    torch.cuda.init()
except Exception as exc:
    sys.exit("%s|%s" % (built, str(exc).replace("\n", " ")))
print("%s|ok" % built)' 2>&1)" || status=$?
  [ "$status" = 3 ] && return 0
  local built="${probe%%|*}" reason="${probe#*|}"
  if [ "$status" = 0 ]; then
    say "torch praat met de kaart (torch is gebouwd tegen CUDA $built)"
    return 0
  fi
  case "$reason" in
    *"too old"*|*"insufficient"*)
      die "de driver op deze pod is te oud voor de torch in deze image. torch is gebouwd tegen CUDA $built en de driver hier levert CUDA $(driver_cuda_version); torch zegt: $reason. Dit is niet op te lossen met vlaggen -- elke vLLM-worker sterft in torch._C._cuda_init(), en met --tensor-parallel-size ziet dat eruit als een worker die omvalt. Twee uitwegen: een image met een torch die bij deze driver past (cu128 bij een 570-driver), of een pod met een nieuwere driver (580+ voor CUDA 13). /workspace blijft staan, dus het model hoeft niet opnieuw gedownload."
      ;;
    *)
      warn "torch kan de kaart hier niet openen: $reason"
      warn "vLLM zal dat ook niet kunnen; dit gaat vrijwel zeker mis bij het opstarten."
      ;;
  esac
}

# What CUDA the driver offers, as nvidia-smi reports it ("CUDA Version: 12.8"),
# or "onbekend". This is the number torch compares against, not the driver
# version itself -- 570.144 and CUDA 12.8 are the same fact said twice.
driver_cuda_version() {
  local found=""
  found="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1 || true)"
  printf '%s' "${found:-onbekend}"
}

# How much shared memory this container has, in megabytes, or empty when the
# question cannot be answered here. vLLM's workers talk to each other over
# /dev/shm, and a container started without --shm-size gets 64 MB.
shm_size_mb() {
  df -BM --output=size /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9' || true
}

# --tensor-parallel-size is a promise about the machine: a worker process per
# card, all of them talking over shared memory. Both ways that promise is
# broken are visible here, in the second before the run starts, and invisible
# afterwards -- vLLM reports either of them as "WorkerProc initialization
# failed due to an exception in a background process", a quarter of an hour of
# rent later and with the model already loaded.
check_tensor_parallel() {
  [ "$TENSOR_PARALLEL" -gt 1 ] || return 0
  if [ "$TENSOR_PARALLEL" -gt "$GPU_COUNT" ]; then
    [ "$FORCE" = 1 ] || die "TENSOR_PARALLEL=$TENSOR_PARALLEL, maar nvidia-smi ziet $GPU_COUNT kaart(en). vLLM start dan een worker per kaart die er niet is en valt tijdens het opstarten om. Zet TENSOR_PARALLEL=$GPU_COUNT, of kijk of CUDA_VISIBLE_DEVICES de andere kaarten wegfiltert. Gebruik --force om toch door te gaan."
    warn "TENSOR_PARALLEL=$TENSOR_PARALLEL terwijl er $GPU_COUNT kaart(en) zijn; doorgaan op eigen risico (--force)"
  fi
  local shm; shm="$(shm_size_mb)"
  if [ -n "$shm" ] && [ "$shm" -lt 1024 ]; then
    warn "/dev/shm is ${shm} MB. De workers van vLLM praten daarover met elkaar en 64 MB (de"
    warn "standaard van een container zonder --shm-size) is te weinig: het opstarten eindigt dan"
    warn "in 'WorkerProc initialization failed'. Start de container met --shm-size 8g of meer."
  fi
}

model_dir() { echo "$HF_HOME/hub/models--${MODEL//\//--}"; }

# Where the tokenizer lives. The harness falls back to the served name when it
# has nothing better, and "qwen3-coder" is not a HuggingFace repo -- so it
# silently estimates context sizes instead of counting them, on a run that is
# entirely about context size. Point it at the snapshot that was downloaded
# anyway; the repo id is the fallback for when the layout surprises us.
tokenizer_source() {
  local snapshot
  # `|| true`: under `set -e` with pipefail a missing snapshots directory would
  # otherwise abort the whole script before the model has been downloaded.
  snapshot="$(find "$(model_dir)/snapshots" -maxdepth 2 -name tokenizer.json -print 2>/dev/null | head -1 || true)"
  if [ -n "$snapshot" ]; then
    dirname "$snapshot"
  else
    echo "$MODEL"
  fi
}

model_size_gb() {
  local dir; dir="$(model_dir)"
  [ -d "$dir" ] || { echo "0"; return; }
  du -sb --dereference "$dir" 2>/dev/null | awk '{printf "%.1f", $1/1024/1024/1024}' | grep . || echo 0
}

# The weights are either there or they are not: 25 GB is the line between a
# finished download and one that only brought the metadata in.
model_is_complete() {
  awk -v s="$(model_size_gb)" 'BEGIN{exit !(s > 25)}'
}

# The directory the weights actually live in, or empty when the download is
# not complete. vLLM takes a path everywhere it takes a repo id, and a path
# needs nothing from the Hub -- which is the way out when the Hub answers 429.
model_snapshot_dir() {
  model_is_complete || return 0
  local config
  # `|| true`: under `set -e` with pipefail a missing snapshots directory
  # would otherwise abort the whole script, while "not there" is an answer.
  config="$(find "$(model_dir)/snapshots" -maxdepth 2 -name config.json -print 2>/dev/null | head -1 || true)"
  if [ -n "$config" ]; then dirname "$config"; fi
  return 0
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
  local tok; tok="$(tokenizer_source)"
  [ -n "$tok" ] && SETS+=(--set "tokenizer.path=$tok")
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
  check_tensor_parallel
  check_driver_matches_torch
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
    # Alleen zeggen wat we hier weten. Dat het pakket er is, betekent nog niet
    # dat er exact geteld wordt: bij de vorige meting was het geinstalleerd en
    # werd er toch geschat, omdat de tokenizer onder de served-model-name werd
    # gezocht en die geen HuggingFace-repo is. `doctor` bouwt de teller echt en
    # zegt het pas als het waar is.
    say "tokenizers aanwezig; doctor controleert straks of er ook exact geteld wordt"
  else
    "$PY" -m pip install tokenizers >&2 \
      || die "tokenizers kon niet geinstalleerd worden. Zonder tokenizer zijn de contextgroottes een schatting; draai eerst 'stresstest calibrate' of los dit op."
  fi
  "$PY" -m pip install matplotlib >&2 || warn "matplotlib niet geinstalleerd; alleen SVG-grafieken (geen bezwaar)"
  [ "$MOCK" = 1 ] || "$PY" -m pip install "huggingface_hub[cli]" hf_transfer >&2
}

# One attempt, so the retry below reads as a retry. Both CLIs resume a partial
# download, so a second attempt costs the bytes that did not arrive yet.
download_model_once() {
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL" >&2
  else
    huggingface-cli download "$MODEL" >&2
  fi
}

download_model() {
  [ "$MOCK" = 1 ] && { say "MOCK: model niet nodig"; return; }
  head_ "Model ophalen: $MODEL"
  local dir size
  dir="$(model_dir)"
  size="$(model_size_gb)"
  if model_is_complete; then
    say "staat er al: $dir (${size} GB), download overgeslagen"
    return
  fi
  [ "$size" != "0" ] && warn "onvolledige download gevonden (${size} GB), wordt hervat"
  # Retrying a download that is not allowed to touch the network costs four
  # attempts and seven minutes before it says the same thing.
  if [ "$HF_OFFLINE" = 1 ]; then
    die "HF_HUB_OFFLINE staat aan, maar $MODEL staat niet compleet op schijf (${size} GB in $dir). Zet HF_HUB_OFFLINE uit voor deze ene stap, of zet de gewichten er zelf neer."
  fi

  mkdir -p "$HF_HOME"
  # huggingface.co rate-limits anonymous requests per IP, and on a rented pod
  # that IP is shared with every other container on the machine: a 429 says
  # nothing about this download and everything about the neighbours. Waiting
  # it out is cheaper than ending a run that has a GPU on the meter.
  local attempt wait_s="$HF_RATE_LIMIT_WAIT_S"
  for attempt in $(seq 1 "$HF_DOWNLOAD_RETRIES"); do
    if download_model_once; then break; fi
    [ "$attempt" -lt "$HF_DOWNLOAD_RETRIES" ] \
      || die "de download van $MODEL bleef mislukken, ook na $HF_DOWNLOAD_RETRIES pogingen. Zie de melding hierboven. Staat daar '429 Too Many Requests', dan knijpt huggingface.co af: wacht een kwartier, of zet een HF_TOKEN in de omgeving -- ingelogd verkeer krijgt een ruimere limiet dan anoniem."
    warn "poging $attempt van $HF_DOWNLOAD_RETRIES mislukt; over ${wait_s}s opnieuw (de download wordt hervat, niet overgedaan)"
    if [ -z "${HF_TOKEN:-}" ]; then
      warn "tip: een HF_TOKEN in de omgeving geeft een ruimere limiet bij huggingface.co dan anoniem verkeer."
    fi
    sleep "$wait_s"
    wait_s=$(( wait_s * 2 > HF_RATE_LIMIT_MAX_WAIT_S ? HF_RATE_LIMIT_MAX_WAIT_S : wait_s * 2 ))
  done

  size="$(model_size_gb)"
  model_is_complete \
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
  # Again: on a clean pod the check above ran before torch existed.
  check_driver_matches_torch
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
# file descriptor, so it lasts exactly as long as some process still has fd 9
# open: a killed run, or a pod stopped mid-run, takes its children with it and
# leaves nothing stale behind.
#
# The children a run waits for -- the harness that does the measuring -- do
# inherit fd 9, and that is the point: kill this shell alone and the harness
# carries on driving the GPU, where a second start is exactly as harmful as
# before and stays refused. The exception is the children that outlive a run by
# design, the deadman above all, which would hold the lock until they fire; so
# every background spawn closes fd 9 with 9>&-.
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
SERVER_ERROR_PATTERNS='no available memory|out of memory|CUDA out of memory|compute capability|not supported|unrecognized arguments|invalid choice|does not exist|Too Many Requests|ValueError|RuntimeError|Exception:|Error'

# What a vLLM log line carries in front of the message: the process that wrote
# it ("(EngineCore pid=1281) ") and the level with its source location
# ("ERROR 09-10 08:13:54 [core.py:1374] "). Both have to come off before the
# patterns above are applied. SERVER_ERROR_PATTERNS ends in a bare "Error",
# which otherwise matches the ERROR prefix of every traceback frame -- and a
# "vermoedelijke oorzaak" of twelve frames ("return func(*args, **kwargs)",
# "^^^^^^", "self._init_executor()") names no cause at all. sed keeps one line
# per line, so grep -n afterwards still counts in the real log.
LOG_PREFIX_STRIP='s/^\([^)]*\) //; s/^(INFO|DEBUG|WARNING|ERROR|CRITICAL)( +[0-9][0-9:.-]* +[0-9][0-9:.]*)? +\[[^]]*\] ?//'

# The scaffolding of a Python traceback, which survives that strip: the frame
# lines and the carets under them. They are the same in every failure.
TRACEBACK_FRAME_PATTERNS='^[0-9]+: *(File "|raise [A-Za-z]|[~^]+ *$)'

# Lines a worker process wrote itself: "(VllmWorker rank=1 pid=1300) ...",
# "(Worker_TP0 pid=...) ...". With --tensor-parallel-size above 1 vLLM runs one
# such process per card, and when one of them dies the API server passes on
# only that it did -- the reason stays in the worker's own lines, hundreds of
# lines above the tail of the log.
WORKER_LOG_PATTERNS='^\([^)]*[Ww]orker[^)]*\)'

# The API server saying exactly that, and nothing more useful than that.
WORKER_FAILURE_PATTERNS='WorkerProc initialization failed|exception in a background process|Failed core proc'

# torch refusing to talk to the driver. This one has to be looked for before
# anything else that a multi-card start can die of: it presents as a worker
# that went away, and a diagnosis about /dev/shm or NCCL would send the
# operator after a machine that is fine.
DRIVER_TOO_OLD_PATTERNS='NVIDIA driver on your system is too old|CUDA driver version is insufficient|no kernel image is available for execution'

# NCCL failing, as opposed to NCCL saying something. Deliberately the error
# names and not a bare "P2P": vLLM logs "custom allreduce is disabled because
# your platform lacks GPU P2P capability" on a perfectly healthy start, and
# spending a start-up on that would cost minutes and measure nothing new.
NCCL_FAILURE_PATTERNS='ncclInternalError|ncclSystemError|ncclUnhandledCudaError|ncclRemoteError|ncclInvalidUsage|NCCL error|DistBackendError|NCCL communicator was aborted|peer access is not supported'

# What the Hub looks like when it is throttling us. Deliberately narrow: not a
# bare "429", which matches a port number or a byte count somewhere in a
# 500-line log, and not vLLM's own "Error retrieving file list", which covers
# every Hub failure including a mistyped MODEL -- waiting one out costs a
# quarter of an hour of GPU rent for nothing.
HF_RATE_LIMIT_PATTERNS='Too Many Requests|429 Client Error|RateLimitExceeded'

# What it looks like when FlashInfer's JIT compiler cannot build for this card.
# The message about sm75 is the symptom: the card is sm_120, and the real reason
# sits a few lines higher ("SM 12.x requires CUDA >= 12.9") -- the toolkit on
# the image is older than what this flashinfer needs for Blackwell. A missing
# module is the other way the same import chain ends.
FLASHINFER_PATTERNS='FlashInfer requires GPUs|check_cuda_arch|SM 12\.x requires CUDA|No module named .flashinfer'

# What says the attention backend is involved, not just the sampler: vLLM
# announcing that it chose FlashInfer, or the traceback frames in which it
# builds FlashInfer's prefill and decode modules (flashinfer/jit/attention/...).
# The sampler goes through flashinfer/sampling.py instead. Not the frame in
# vllm/v1/attention/backends/flashinfer.py: the sampler imports that module
# too, so it shows up in a sampler-only failure as well.
FLASHINFER_ATTENTION_PATTERNS='Using FlashInfer backend|flashinfer/jit/attention|gen_customize_batch_(prefill|decode)_module|gen_batch_(prefill|decode)_module'

# The one way the tool-call flags themselves are refused. Not a bare
# "tool.call.parser": vLLM echoes every non-default argument at start-up
# ("non-default args: {... 'tool_call_parser': 'qwen3_coder' ...}"), so that
# would match every log, and a server that died on something else would be
# restarted without tool calling -- and measured that way.
TOOL_PARSER_ERROR_PATTERNS='invalid tool call parser|tool-call-parser: invalid choice|tool.call.parser.*not (found|supported|registered)|unrecognized arguments:.*(--tool-call-parser|--enable-auto-tool-choice)|enable-auto-tool-choice requires'

# The lines of the log that name a reason, with the log's own prefixes taken
# off so that a traceback frame cannot pass for one.
log_suspects() {
  sed -E "$LOG_PREFIX_STRIP" "$1" 2>/dev/null \
    | grep -nEi "$SERVER_ERROR_PATTERNS" \
    | grep -vE "$TRACEBACK_FRAME_PATTERNS" \
    | tail -"${2:-12}" || true
}

# What the workers said, minus the frames: their exception is the root cause
# the API server refers to and does not repeat.
worker_lines() {
  grep -nE "$WORKER_LOG_PATTERNS" "$1" 2>/dev/null \
    | grep -vE 'File "[^"]*", line [0-9]+|[~^]{3,} *$' \
    | tail -"${2:-15}" || true
}

show_server_error() {
  local log="$1" hits workers
  [ -f "$log" ] || return 0
  hits="$(log_suspects "$log")"
  if [ -n "$hits" ]; then
    printf '%s\n' "--- vermoedelijke oorzaak, uit $log ---" >&2
    printf '%s\n' "$hits" >&2
  fi
  # "See stack trace for root cause" -- that stack trace is not below this
  # point in the log, it is in another process's lines further up, so neither
  # the summary above nor the tail below reaches it.
  if grep -qE "$WORKER_FAILURE_PATTERNS" "$log" 2>/dev/null; then
    workers="$(worker_lines "$log")"
    if [ -n "$workers" ]; then
      printf '%s\n' "--- wat de workers zelf zeiden (de oorzaak waar de API-server naar verwijst) ---" >&2
      printf '%s\n' "$workers" >&2
    fi
  fi
  printf '%s\n' "--- laatste 25 regels van $log ---" >&2
  tail -n 25 "$log" >&2 || true
}

# Why the server is not there, in the operator's words. Shared by the two ways
# a start-up runs out of options: an attempt with nothing left to try, and a
# loop that has spent every attempt.
#   die_server_start <status from wait_ready>
die_server_start() {
  local status="${1:-2}"
  # First, because it is the one cause that has nothing to do with this
  # machine -- and the generic message below would send the operator after the
  # KV cache for a problem the neighbours' traffic caused.
  if grep -qEi "$HF_RATE_LIMIT_PATTERNS" "$SERVER_LOG" 2>/dev/null; then
    die "huggingface.co blijft vLLM met 429 (te veel verzoeken) afwijzen. Dit is geen geheugen- of kernelprobleem: vLLM vraagt bij elke start de bestandslijst op bij de Hub, ook als het model al op schijf staat, en die limiet per IP-adres deel je met alle andere containers op deze machine. Uitwegen: haal het model eerst compleet binnen ('scripts/pod.sh setup'), want dan dient dit script de map zelf aan vLLM aan in plaats van de repo-naam; of zet HF_HUB_OFFLINE=1; of zet een HF_TOKEN in de omgeving en probeer het over een kwartier opnieuw."
  fi
  if [ "$status" = 2 ]; then
    # Before the rest: this dies in every worker, at CUDA initialisation, and
    # so wears the clothes of whatever executor was starting them.
    if grep -qE "$DRIVER_TOO_OLD_PATTERNS" "$SERVER_LOG" 2>/dev/null; then
      die "torch kan de kaart niet openen: de driver op deze pod is te oud voor de torch in deze image (zie hierboven, 'The NVIDIA driver on your system is too old'). De driver hier levert CUDA $(driver_cuda_version). Geen vlag helpt hiertegen -- ook TENSOR_PARALLEL=1 niet, het is alleen minder zichtbaar. Uitwegen: een image met een torch die bij deze driver past (cu128 bij een 570-driver), of een pod met een nieuwere driver (580+ voor CUDA 13). /workspace blijft staan, dus het model hoeft niet opnieuw gedownload."
    fi
    if grep -qE 'unrecognized arguments:.*--attention-backend' "$SERVER_LOG" 2>/dev/null; then
      die "deze vLLM kent --attention-backend niet (een oudere vLLM dan de 0.28 waar dit script op is afgestemd), en zonder die vlag is er geen weg om FlashInfer heen. Werk vLLM bij, of kies een image met een recente vLLM en een CUDA-toolkit van 12.9 of nieuwer."
    fi
    if grep -q "No module named .flashinfer" "$SERVER_LOG" 2>/dev/null; then
      die "vLLM importeert flashinfer ook als hij het niet gebruikt, en het pakket is hier weg. Zet het terug ( pip install flashinfer-python ) en start opnieuw; dit script zet zelf VLLM_USE_FLASHINFER_SAMPLER=0 en --attention-backend TRITON_ATTN zodat de JIT-compiler er niet aan te pas komt."
    fi
    if grep -qEi 'FlashInfer requires GPUs|check_cuda_arch|SM 12\.x requires CUDA' "$SERVER_LOG" 2>/dev/null; then
      die "vLLM blijft op FlashInfer stuklopen, ook zonder de FlashInfer-sampler en met --attention-backend TRITON_ATTN (zie hierboven welke van de twee dit script al geprobeerd heeft). De JIT-compiler van FlashInfer kan deze kaart (sm_120) niet bouwen met de CUDA-toolkit in deze image; het log noemt CUDA >= 12.9. Kijk in de traceback welk onderdeel van vLLM hem nu nog aanroept. Een image met een toolkit van 12.9 of nieuwer is de zekere uitweg -- /workspace blijft staan, dus het model hoeft niet opnieuw gedownload."
    fi
    # More than one card means one worker process per card, and the API
    # server's own traceback is then about the death of a process it was
    # waiting for -- never about the reason. Naming the three things that
    # actually break here saves the operator a spelunk through the log, and
    # /dev/shm in particular is invisible unless you go and look: the default
    # of 64 MB in a container is far too little for vLLM's worker channels.
    if [ "${TENSOR_PARALLEL:-1}" -gt 1 ] \
       && grep -qE "$WORKER_FAILURE_PATTERNS" "$SERVER_LOG" 2>/dev/null; then
      local shm nccl_tried; shm="$(shm_size_mb)"
      nccl_tried=""
      [ "${NCCL_P2P_OFF:-0}" = 1 ] && nccl_tried=", en heeft dat hier al gedaan"
      die "vLLM's workers kwamen niet omhoog. Met --tensor-parallel-size ${TENSOR_PARALLEL} draait er een proces per kaart, en de API-server meldt alleen dat er een is omgevallen; de reden staat hierboven onder 'wat de workers zelf zeiden'. Dit zijn de drie die het meestal zijn: (1) te weinig gedeeld geheugen -- /dev/shm is hier ${shm:-onbekend} MB en daar praten de workers met elkaar; start de container met --shm-size 8g of meer. (2) de kaarten bereiken elkaar niet (NCCL); NCCL_P2P_DISABLE=1 zet de directe kaart-tot-kaart-weg uit en is vaak genoeg -- dit script probeert dat zelf zodra het log NCCL noemt${nccl_tried}. (3) er zijn ${GPU_COUNT:-?} kaarten zichtbaar voor ${TENSOR_PARALLEL} processen; kijk wat nvidia-smi en CUDA_VISIBLE_DEVICES zeggen. TENSOR_PARALLEL=1 komt wel omhoog, maar meet een andere opstelling: leg het vast als je dat doet."
    fi
    die "vLLM is tijdens het opstarten gestopt. Zie hierboven en $SERVER_LOG. Vaakst voorkomend: te weinig geheugen voor de KV-cache (verlaag --max-model-len of GPU_UTIL), of een vLLM zonder kernels voor deze kaart."
  fi
  die "vLLM kwam niet omhoog binnen ${SERVER_START_TIMEOUT_S}s, maar draait nog wel. Zie $SERVER_LOG; verhoog zo nodig SERVER_START_TIMEOUT_S."
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
  local server_env attempt model_arg snapshot
  local hf_wait_s="$HF_RATE_LIMIT_WAIT_S"
  for attempt in $(seq 1 "$SERVER_START_ATTEMPTS"); do
    # A path where a repo id would do. Identical weights and identical config,
    # so the measurement is the same; the difference is that vLLM then needs
    # nothing from huggingface.co to start.
    model_arg="$MODEL"
    if [ "$HF_OFFLINE" = 1 ]; then
      snapshot="$(model_snapshot_dir)"
      [ -n "$snapshot" ] && model_arg="$snapshot"
    fi
    local cmd="vllm serve $model_arg --served-model-name $SERVED_NAME --host $HOST_BIND --port $PORT"
    cmd="$cmd --gpu-memory-utilization $GPU_UTIL --enable-prefix-caching $tunable"
    [ "$TENSOR_PARALLEL" -gt 1 ] && cmd="$cmd --tensor-parallel-size $TENSOR_PARALLEL"
    # As a server argument, not VLLM_ATTENTION_BACKEND: vLLM 0.28 no longer
    # reads that variable and only warns about it.
    [ -n "$backend" ] && cmd="$cmd --attention-backend $backend"
    cmd="$cmd $tool_flags"

    say "start: $cmd"
    server_env=(HF_HOME="$HF_HOME")
    if [ "$HF_OFFLINE" = 1 ]; then
      server_env+=(HF_HUB_OFFLINE=1)
      say "       HF_HUB_OFFLINE=1 (alles komt van schijf, niets van huggingface.co)"
    fi
    if [ "$NO_FLASHINFER_SAMPLER" = 1 ]; then
      server_env+=(VLLM_USE_FLASHINFER_SAMPLER=0)
      say "       VLLM_USE_FLASHINFER_SAMPLER=0"
    fi
    if [ "$NCCL_P2P_OFF" = 1 ]; then
      server_env+=(NCCL_P2P_DISABLE=1)
      say "       NCCL_P2P_DISABLE=1 (de kaarten praten via het werkgeheugen, niet rechtstreeks)"
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
      echo "$NCCL_P2P_OFF" > "$STATE_DIR/current_nccl_p2p"
      printf '%s' "$tunable" | sed -n 's/.*--kv-cache-dtype \([a-z0-9]*\).*/\1/p' \
        > "$STATE_DIR/current_kv_dtype"
      return 0
    fi

    # The Hub throttles anonymous API calls per IP, and vLLM asks it for the
    # repo's file list on every start -- also when all 31 GB are already on
    # disk. A 429 there stops a server that needs nothing from the network,
    # and the log blames neither memory nor kernels: it is the traffic of the
    # other containers on the same machine.
    if [ "$status" = 2 ] \
       && grep -qEi "$HF_RATE_LIMIT_PATTERNS" "$SERVER_LOG" 2>/dev/null; then
      snapshot="$(model_snapshot_dir)"
      if [ "$HF_OFFLINE" != 1 ] && [ -n "$snapshot" ]; then
        warn "huggingface.co antwoordt met 429 (te veel verzoeken), terwijl het model compleet op schijf staat."
        warn "opnieuw vanaf die map ($snapshot), zonder de Hub te raadplegen. Dat raakt de meting niet: het zijn dezelfde gewichten."
        HF_OFFLINE=1
        continue
      fi
      if [ "$HF_OFFLINE" != 1 ]; then
        warn "huggingface.co antwoordt met 429 (te veel verzoeken) en er staat nog geen complete download ($(model_size_gb) GB)."
        warn "wachten ${hf_wait_s}s en dan opnieuw; haal het model anders eerst apart binnen met 'scripts/pod.sh setup'."
        if [ -z "${HF_TOKEN:-}" ]; then
          warn "tip: een HF_TOKEN in de omgeving geeft een ruimere limiet dan anoniem verkeer."
        fi
        sleep "$hf_wait_s"
        hf_wait_s=$(( hf_wait_s * 2 > HF_RATE_LIMIT_MAX_WAIT_S ? HF_RATE_LIMIT_MAX_WAIT_S : hf_wait_s * 2 ))
        continue
      fi
    fi

    # FlashInfer's JIT compiler refuses to build here: the log says "Failed to
    # get device capability: SM 12.x requires CUDA >= 12.9", after which
    # check_cuda_arch() reports the misleading "FlashInfer requires GPUs with
    # sm75 or higher" -- the card is sm_120. The toolkit on the image is simply
    # older than what this flashinfer needs for Blackwell.
    #
    # vLLM reaches for FlashInfer in two places, and both have to be steered
    # around it. The top-k/top-p sampler uses it whatever the attention
    # backend is; vLLM can sample without it, and that changes nothing about
    # what is measured -- the harness runs at temperature 0 -- so it is safe
    # for the engine variants too. And with --kv-cache-dtype fp8 vLLM picks
    # FLASHINFER as the attention backend, because FLASH_ATTN cannot serve an
    # FP8 cache on this card; the JIT then dies building the prefill module
    # (flashinfer/jit/attention/...) before the sampler is even reached. Only
    # TRITON_ATTN remains, and that one vLLM names itself: "out of potential
    # backends: ['FLASHINFER', 'TRITON_ATTN']". A different backend does
    # change the numbers, so it is announced, made sticky for the rest of the
    # run, and recorded with the results as hardware.attention_backend.
    #
    # Which of the two the log blames decides the order: a traceback through
    # the attention modules means the sampler alone will not help, so both
    # are switched at once rather than spending a start-up on each. A backend
    # the operator chose is never substituted; then only the sampler is tried.
    if [ "$status" = 2 ] && grep -qEi "$FLASHINFER_PATTERNS" "$SERVER_LOG" 2>/dev/null; then
      if [ -z "$backend" ] \
         && grep -qEi "$FLASHINFER_ATTENTION_PATTERNS" "$SERVER_LOG" 2>/dev/null; then
        warn "vLLM koos FlashInfer als attention-backend en de JIT-compiler kan die niet bouwen voor deze kaart;"
        warn "opnieuw met --attention-backend TRITON_ATTN (en zonder de FlashInfer-sampler)."
        warn "Dat is een andere backend en dus andere getallen: dit wordt bij de resultaten vastgelegd."
        ATTENTION_BACKEND="TRITON_ATTN"
        backend="$ATTENTION_BACKEND"
        NO_FLASHINFER_SAMPLER=1
        continue
      fi
      if [ "$NO_FLASHINFER_SAMPLER" != 1 ]; then
        warn "vLLM struikelt over FlashInfer; opnieuw zonder de FlashInfer-sampler."
        warn "Dat raakt de meting niet: het harnas draait op temperatuur 0."
        NO_FLASHINFER_SAMPLER=1
        continue
      fi
      if [ -z "$backend" ]; then
        warn "vLLM struikelt ook zonder de FlashInfer-sampler nog over FlashInfer;"
        warn "opnieuw met --attention-backend TRITON_ATTN. Dat is een andere backend en dus"
        warn "andere getallen: dit wordt bij de resultaten vastgelegd."
        ATTENTION_BACKEND="TRITON_ATTN"
        backend="$ATTENTION_BACKEND"
        continue
      fi
      # Sampler off and a non-FlashInfer backend, and still FlashInfer: this
      # is nothing the script can steer around. die_server_start says so.
    fi

    # Two cards that cannot reach each other directly. NCCL wants the PCIe
    # peer-to-peer path and consumer Blackwell does not always offer it; the
    # worker then dies where it opens the communicator, and the API server
    # passes on only that a background process went away. NCCL_P2P_DISABLE=1
    # routes the collectives over host memory instead: slower per exchange,
    # but it comes up. Slower is a different measurement, so it is announced
    # here, made sticky for the rest of the run, and left in the log.
    if [ "$status" = 2 ] && [ "$TENSOR_PARALLEL" -gt 1 ] && [ "$NCCL_P2P_OFF" != 1 ] \
       && grep -qEi "$NCCL_FAILURE_PATTERNS" "$SERVER_LOG" 2>/dev/null; then
      warn "NCCL komt niet door tussen de kaarten; opnieuw met NCCL_P2P_DISABLE=1, dan gaat het"
      warn "verkeer via het werkgeheugen in plaats van rechtstreeks over PCIe."
      warn "Dat is trager voor alles wat de kaarten uitwisselen: noteer het bij de resultaten."
      NCCL_P2P_OFF=1
      continue
    fi

    # The README's documented fallback, but only when the log actually blames
    # the tool-call parser. Retrying blindly costs another start-up and, worse,
    # points the operator at the wrong thing: an engine that dies on memory or
    # on missing kernels dies again in exactly the same way.
    if [ "$status" = 2 ] && [ -n "$tool_flags" ] \
       && grep -qEi "$TOOL_PARSER_ERROR_PATTERNS" "$SERVER_LOG" 2>/dev/null; then
      warn "de tool-call-parser wordt niet geaccepteerd; opnieuw zonder die twee vlaggen"
      warn "(het harnas schakelt dan zelf over op een tekstvariant met dezelfde berichtstructuur)"
      tool_flags=""
      continue
    fi

    show_server_error "$SERVER_LOG"
    if [ "$strict" = 1 ]; then
      return 1
    fi
    die_server_start "$status"
  done

  # Every attempt spent. Falling out of the loop must not read as success:
  # start_server would return 0 and the harness would go and measure a server
  # that is not there, which shows up as a wall of connection errors much
  # later.
  show_server_error "$SERVER_LOG"
  if [ "$strict" = 1 ]; then
    return 1
  fi
  warn "$attempt pogingen gedaan, en vLLM staat nog steeds niet."
  die_server_start "${status:-2}"
}

# Counting series is too crude: what matters is whether the three numbers the
# main question hangs on are actually exposed. Without preemptions, prefix
# cache hits and KV occupancy you are only measuring latency, which is half
# the question.
# The API server answers /v1/models while the engine is still registering its
# metrics, so a scrape one second later can miss vllm:num_preemptions_total and
# read as "the endpoint is broken" when nothing is wrong. Seen on a first start
# that took two minutes; the same server had the series moments later, at 0.
#
# Hence the retries. The warm-up request in front of them is worth its second
# on its own: until now the check only touched /v1/models, so nothing had
# proved that the model actually generates before the run started.
warm_up_server() {
  curl -s -m 120 -o /dev/null \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",\"max_tokens\":8,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"hallo\"}]}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" || true
}

check_metrics() {
  local body count missing="" attempt
  warm_up_server
  for attempt in 1 2 3; do
    body="$(curl -s "http://127.0.0.1:$PORT/metrics" || true)"
    printf '%s' "$body" | grep -q '^vllm:num_preemptions' && break
    [ "$attempt" = 3 ] || sleep 5
  done
  count="$(printf '%s' "$body" | grep -c '^vllm:' || true)"

  printf '%s' "$body" | grep -q '^vllm:\(num_preemptions\)' || missing="$missing preempties"
  printf '%s' "$body" | grep -q '^vllm:\(gpu_\)\?prefix_cache_\(queries\|hits\|hit_rate\)' || missing="$missing prefix-cache"
  printf '%s' "$body" | grep -q '^vllm:\(kv_cache_usage_perc\|gpu_cache_usage_perc\)' || missing="$missing KV-bezetting"

  if [ -n "$missing" ]; then
    [ "$FORCE" = 1 ] || die "/metrics mist:$missing (van $count vllm-reeksen), ook na een opwarmverzoek. Daarmee is de hoofdvraag niet te beantwoorden -- zonder preempties, cache hit rate en KV-bezetting meet je alleen latentie. Zet de Prometheus-endpoint aan, of gebruik --force."
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
#
# The harness does the counting, because it also has to answer the question
# underneath it: were the runs already in this directory measured with the
# setup we are running now? A run id says students and context and nothing
# about the corpus or the behaviour model, so without that check a resume
# quietly mixes two measurements. It exits 3 when they differ, and the caller
# turns that into a stop -- carrying on would produce a report that looks
# complete and is not.
pending_ids() {
  local group="$1" dir="$2"
  local flags=()
  [ "$RESUME_ANYWAY" = 1 ] && flags+=(--resume-anyway)
  harness_sets
  "$PY" -m stresstest pending "$dir" --only "$group" \
        "${flags[@]+"${flags[@]}"}" "${SETS[@]}"
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
  pending="$(pending_ids "$group" "$dir")" \
    || die "kan groep $group niet hervatten in $dir -- zie de melding hierboven."
  if [ -z "$pending" ]; then
    say "groep $group: alles staat al in $dir, overslaan"
    return 0
  fi
  head_ "Groep $group ($(echo "$pending" | wc -w | tr -d ' ') runs)"
  harness_sets
  local resume=()
  [ "$RESUME_ANYWAY" = 1 ] && resume+=(--resume-anyway)
  # shellcheck disable=SC2086 -- word splitting of the id list is intended
  "$PY" -m stresstest matrix --only "$group" --out "$dir" --run $pending --no-pause \
        "${resume[@]+"${resume[@]}"}" "${SETS[@]}"
}

# The engine variants cannot be driven from the client: each one needs its own
# vLLM. The wrapper restarts the server itself, which is why --no-pause is
# correct here and meaningless without it.
run_engine_group() {
  local dir="$1" name flags context need line pending
  head_ "Engine-varianten (elke variant herstart vLLM)"
  # Asked once, before the first restart: which variants are still missing, and
  # was this directory measured with the setup we are running now? The engine
  # settings themselves are exempt -- varying those is what this group does --
  # but the corpus and the behaviour model are not.
  pending="$(pending_ids engine "$dir")" \
    || die "kan de engine-varianten niet hervatten in $dir -- zie de melding hierboven."
  while IFS=$'\t' read -r name flags context; do
    [ -n "$name" ] || continue
    case " $pending " in
      *" engine_$name "*) : ;;
      *) say "engine_$name staat al in $dir, overslaan"; continue ;;
    esac
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
    local resume=()
    [ "$RESUME_ANYWAY" = 1 ] && resume+=(--resume-anyway)
    "$PY" -m stresstest matrix --run "engine_$name" --only engine --out "$dir" --no-pause \
          "${resume[@]+"${resume[@]}"}" "${SETS[@]}"
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

# Watching a server that a real class is loading, instead of making the load
# ourselves. Deliberately does not touch vLLM: it must be safe to start, stop
# and restart while twenty students are working.
cmd_monitor() {
  detect_gpu
  harness_sets
  local dir="${1:-}"
  [ -n "$dir" ] || dir="results/$(date +%Y%m%d-%H%M%S)_monitor"
  shift 2>/dev/null || true
  say "meet de draaiende server; Ctrl-C stopt en schrijft de samenvatting"
  "$PY" -m stresstest monitor --out "$dir" "${SETS[@]}" "$@"
}

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

# Can we actually push? Asked before the measurement, not after it.
#
# The pod only stops once the results are pushed, so broken push access means
# discovering after six hours that the machine is still running and the results
# are still on it.
#
# Reading is not writing. `git ls-remote` succeeds with a `contents: read`
# token, with a token whose write scope was revoked, and against a repository
# whose branch protection refuses new branches -- all three used to pass this
# check and fail after the whole measurement. So the check pushes: a throwaway
# ref, deleted straight away. Nothing else proves write access without
# guessing.
#
# `ls-remote` stays as the first, cheap step, because "no network" and "network
# but no write access" need different answers.
check_push_access() {
  [ "$PUSH_RESULTS" = 1 ] || return 0
  [ "$MOCK" = 1 ] && return 0
  head_ "Controle: mag deze pod pushen?"
  cd "$REPO_DIR"
  local remote
  if ! remote="$(git remote get-url origin 2>/dev/null)"; then
    warn "geen remote 'origin'."
    _push_access_hint
    return 1
  fi
  setup_git_credentials "$remote" || return 1
  if ! git ls-remote origin >/dev/null 2>&1; then
    warn "kan de remote niet bereiken: $remote"
    warn "Dat is een netwerk- of tokenprobleem, nog voor er van pushen sprake is."
    _push_access_hint
    return 1
  fi

  local probe="push-probe-$$-$(date +%s)"
  if ! git push --quiet origin "HEAD:refs/heads/$probe" 2>/dev/null; then
    warn "de remote is bereikbaar, maar deze pod mag er niet naartoe pushen."
    warn "($remote -- lezen lukt, schrijven niet: een token met alleen"
    warn " leesrechten, een ingetrokken schrijfrecht, of branch protection"
    warn " die nieuwe branches weigert.)"
    _push_access_hint
    return 1
  fi
  if ! git push --quiet origin --delete "$probe" 2>/dev/null; then
    warn "de proefbranch '$probe' is aangemaakt maar niet opgeruimd; verwijder"
    warn "hem zelf:  git push origin --delete $probe"
  fi
  say "push-toegang in orde ($remote)"
  return 0
}

_push_access_hint() {
  warn "De pod stopt pas als de resultaten gepusht zijn, dus dit is nu een"
  warn "probleem en niet over zes uur. Drie manieren om het op te lossen:"
  warn "  1. Token via een RunPod-secret. Maak het secret aan, en zet in je"
  warn "     Pod-template een omgevingsvariabele:"
  warn "         GITHUB_TOKEN = {{ RUNPOD_SECRET_<naam-van-je-secret> }}"
  warn "     De variabele MOET GITHUB_TOKEN of GH_TOKEN heten; de naam van het"
  warn "     secret zelf maakt niet uit. Een secret komt niet vanzelf in de"
  warn "     omgeving -- die verwijzing in de template is wat hem er zet."
  warn "  2. Eenmalig in deze shell:  export GITHUB_TOKEN=..."
  warn "  3. Niet pushen:             scripts/pod.sh all --no-push"
  warn "     (dan stopt de pod ook niet vanzelf, en haal je de resultaten"
  warn "      zelf op met scp)"
}

# Where the push credential comes from. A token in the remote URL ends up in
# .git/config in plain text and in every `git remote -v`; a token in argv ends
# up in the process list. GIT_ASKPASS keeps it in neither: git asks the helper
# script, the script reads the environment, and nothing is written down.
#
# This is what makes a RunPod secret work end to end: store the token as a
# secret, reference it in the template as
#   GITHUB_TOKEN={{ RUNPOD_SECRET_<naam> }}
# and the pod pushes without anyone typing it.
setup_git_credentials() {
  local remote="$1"
  local token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

  case "$remote" in
    *@*) return 0 ;;   # credentials already in the URL, or an ssh remote
  esac
  if [ -z "$token" ]; then
    # No token anywhere. Not fatal: an ssh remote or a credential helper the
    # operator set up themselves works fine, and the push will say so if not.
    return 0
  fi
  case "$remote" in
    https://*) : ;;
    *) return 0 ;;     # a token is no use to a non-https remote
  esac

  local helper="$STATE_DIR/git-askpass.sh"
  mkdir -p "$STATE_DIR"
  # The token is never written to this file, only read from the environment
  # when git calls it.
  cat > "$helper" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  Username*) echo "x-access-token" ;;
  *)         echo "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ;;
esac
ASKPASS
  chmod 700 "$helper"
  export GIT_ASKPASS="$helper"
  export GIT_TERMINAL_PROMPT=0
  say "push-token uit de omgeving (GITHUB_TOKEN); niet opgeslagen in .git/config"
  return 0
}

# results/ is in .gitignore on purpose: local rehearsals should not end up in
# the repository. A measurement that cost real money should, so this adds it
# with -f -- exactly as the comment in .gitignore prescribes.
push_results() {
  head_ "Resultaten naar de repo pushen"
  command -v git >/dev/null 2>&1 || { warn "git ontbreekt op deze pod; niets gepusht."; return 1; }
  cd "$REPO_DIR"
  git rev-parse --git-dir >/dev/null 2>&1 \
    || { warn "$REPO_DIR is geen git-repo; niets gepusht."; return 1; }
  local remote
  if ! remote="$(git remote get-url origin 2>/dev/null)"; then
    warn "geen remote 'origin'. Zet er een, of geef een token mee via de omgeving:"
    warn "    git remote add origin https://github.com/<eigenaar>/<repo>.git"
    warn "    GITHUB_TOKEN=...  (bij RunPod: Secrets, dan {{ RUNPOD_SECRET_<naam> }} in de template)"
    return 1
  fi

  # A rented pod has no git identity, and a commit without one fails.
  git config user.email >/dev/null 2>&1 || git config user.email "pod@stresstest.local"
  git config user.name  >/dev/null 2>&1 || git config user.name  "stresstest pod"

  setup_git_credentials "$remote" || return 1

  local slug branch
  slug="$(printf '%s' "${GPU_NAME:-gpu}" | tr -c '[:alnum:]' '-' | tr -s '-' | sed 's/^-//;s/-$//')"
  branch="${RESULTS_BRANCH:-resultaten/${slug:-gpu}-$(date +%Y%m%d-%H%M%S)}"
  git checkout -B "$branch" >/dev/null 2>&1 || { warn "kon branch $branch niet maken"; return 1; }

  local staged=0 d
  for d in "$@"; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    git add -f "$d" && staged=1
  done
  # The combined report over both phases -- only when this run wrote it. The
  # file survives between runs, so staging it unconditionally pushes a report
  # of an earlier measurement as part of this one.
  if [ "${REPORT_WRITTEN:-0}" = 1 ] && [ -f RESULTATEN.md ]; then
    git add -f RESULTATEN.md
  elif [ -f RESULTATEN.md ]; then
    say "RESULTATEN.md in de hoofdmap is niet door deze run geschreven en wordt niet meegepusht"
  fi
  if [ "$staged" = 0 ]; then
    warn "geen resultatenmappen gevonden om te pushen."
    return 1
  fi
  if git diff --cached --quiet; then
    say "niets nieuws om te committen -- alles stond er al in."
  else
    git commit -q -m "Meetresultaten van ${GPU_NAME:-onbekende kaart}

Gedraaid door scripts/pod.sh op $(date -u '+%Y-%m-%dT%H:%M:%SZ').
Mappen: $*" || { warn "commit mislukt"; return 1; }
  fi

  # Network on a rented pod is not always immediately willing; a few tries with
  # a growing pause costs nothing and saves the whole run.
  local try=1 wait=2
  while :; do
    if git push -u origin "$branch" >&2; then
      say "gepusht naar branch $branch"
      return 0
    fi
    if [ "$try" -ge "$PUSH_RETRIES" ]; then
      warn "push mislukt na $try pogingen. De resultaten staan wel in de commit"
      warn "op branch $branch; push hem handmatig, of haal hem op met scp."
      return 1
    fi
    warn "push mislukt (poging $try van $PUSH_RETRIES), opnieuw over ${wait}s"
    sleep "$wait"
    try=$((try + 1)); wait=$((wait * 2))
  done
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

# The lesson directory from an earlier run, but only if it belongs with this
# measurement. With --skip-lesson there is no phase 2 in this run, and folding
# in an arbitrary older one is how a report ends up holding two behaviour
# models without saying so (issue #20). Same fingerprint or nothing.
reusable_lesson_dir() {
  local matrix_dir="$1" previous
  previous="$(cat "$STATE_DIR/last_lesson_dir" 2>/dev/null || true)"
  if [ -z "$previous" ] || [ ! -d "$previous" ]; then
    say "--skip-lesson: geen eerdere lesvalidatie bekend om bij te voegen"
    return 0
  fi
  if "$PY" -m stresstest fingerprint --same "$matrix_dir" "$previous" >/dev/null 2>&1; then
    say "--skip-lesson: eerdere lesvalidatie $previous hoort bij deze opstelling en wordt meegenomen"
    echo "$previous"
    return 0
  fi
  say "--skip-lesson: eerdere lesvalidatie $previous is met een andere opstelling"
  say "  gemeten (of draagt geen vingerafdruk) en wordt niet meegenomen"
  return 0
}

plural_runs() { [ "$1" = 1 ] && printf '1 run' || printf '%s runs' "$1"; }

# How old the directory is and what is already in it. Resuming is the point of
# the directory, but resuming in yesterday's measurement without noticing is
# how a run finishes in ten minutes having measured nothing (issue #18). The
# fingerprint check refuses that; this line makes it visible even when the
# fingerprint does match.
describe_results_dir() {
  local dir="$1" stamp epoch age runs
  # The `|| true` sits inside the substitution, next to find, not after the
  # assignment. A resumed directory whose first server never came up has no
  # runs/ yet; find then exits 1, pipefail hands that on, and the ERR trap
  # ends this line -- a directory listing -- with "afgebroken op regel N",
  # twice, and the description the operator was about to read comes out empty.
  runs="$( { find "$dir/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null || true; } | wc -l | tr -d " ")"
  # The directory name carries the moment it was created: 20260908-180708_matrix.
  stamp="$(basename "$dir" | sed -n 's/^\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)-\([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\).*/\1-\2-\3 \4:\5:\6/p')"
  epoch=""
  [ -n "$stamp" ] && epoch="$(date -d "$stamp" +%s 2>/dev/null || true)"
  [ -n "$epoch" ] || epoch="$(date -r "$dir" +%s 2>/dev/null || true)"
  if [ -n "$epoch" ]; then
    age="$(awk -v s="$(( $(date +%s) - epoch ))" 'BEGIN{
      if (s < 5400) printf "%d minuten", s/60; else printf "%d uur", s/3600}')"
    printf 'aangemaakt %s geleden, %s aanwezig' "$age" "$(plural_runs "$runs")"
  else
    printf '%s aanwezig' "$(plural_runs "$runs")"
  fi
}

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
      say "hervat in bestaande map $dir ($(describe_results_dir "$dir"))"
      say "  zet RESULTS_DIR= om ergens anders te beginnen"
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

  if ! check_push_access; then
    [ "$FORCE" = 1 ] || die "geen push-toegang. Zie de aanwijzingen hierboven, of draai met --no-push."
    warn "geen push-toegang, doorgaan vanwege --force"
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
  else
    lesson_dir="$(reusable_lesson_dir "$dir")"
  fi

  # One report covering both phases, next to the per-phase ones.
  if [ -n "$lesson_dir" ]; then
    if "$PY" -m stresstest report "$dir" --also "$lesson_dir" --out RESULTATEN.md; then
      REPORT_WRITTEN=1
    else
      warn "gecombineerd rapport mislukt; de losse rapporten staan er wel"
    fi
  else
    # RESULTATEN.md in the repository root is the file that carries the budget
    # request. Leaving yesterday's version there and pushing it along suggests
    # this measurement covers phase 2, which it does not.
    warn "deze meting heeft geen bijbehorende lesvalidatie, dus RESULTATEN.md in"
    warn "de hoofdmap is niet bijgewerkt en wordt niet meegepusht: hij hoort bij"
    warn "een eerdere meting. Fase 1 staat compleet in $dir/RESULTATEN.md."
    warn "Alsnog combineren, zodra er een passende lesmap is:"
    warn "    $PY -m stresstest report $dir --also <lesmap> --out RESULTATEN.md"
  fi

  head_ "Klaar in $(awk -v s=$((SECONDS - started)) 'BEGIN{printf "%du%02dm", s/3600, (s%3600)/60}')"
  say "fase 1: $dir/RESULTATEN.md"
  [ -n "$lesson_dir" ] && say "fase 2: $lesson_dir/RESULTATEN.md"
  archive_results >/dev/null || true
  echo >&2
  say "Haal de resultaten op, vanaf je eigen machine:"
  say "    scp -P <poort> -i <private-sleutel> -r root@<ip>:$REPO_DIR/results ./results-van-de-gpu"
  say "    De poort staat bij RunPod onder Connect > SSH over exposed TCP -- niet 22,"
  say "    en de proxy-SSH erboven kan geen scp. Zie VALKUILEN.md."
  echo >&2
  warn "ZET DE INSTANCE UIT als je klaar bent. Terminate, niet Stop -- bij Stop tikt de opslag door."

  local pushed=0
  if [ "$PUSH_RESULTS" = 1 ]; then
    push_results "$dir" "$lesson_dir" && pushed=1
  fi

  if [ "$pushed" = 1 ] || [ "$SHUTDOWN_WHEN_DONE" = 1 ]; then
    if [ "$pushed" = 1 ]; then
      say "de resultaten staan in de repo; de pod wordt over 10 minuten gestopt."
    else
      warn "de pod wordt over 10 minuten gestopt (--shutdown)."
    fi
    warn "Afbreken: scripts/pod.sh disarm"
    arm_deadman 0.17
  elif [ "$PUSH_RESULTS" = 1 ]; then
    # The push is what makes the results safe. Without it the pod must not stop
    # -- but it must not run forever either, so whatever deadman was armed at
    # the start stays armed.
    warn "de resultaten zijn NIET gepusht; de pod blijft draaien zodat je ze kunt ophalen."
    if [ -f "$DEADMAN_PID_FILE" ]; then
      warn "de doodsklok blijft staan ($(cat "$STATE_DIR/deadman_at" 2>/dev/null || echo '?'))."
    else
      warn "er staat geen doodsklok. Zet er een:  scripts/pod.sh deadman 2"
    fi
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
  monitor [map]      een draaiende server meten zonder zelf belasting te
                     maken: voor een echte les via OpenCode. Ctrl-C stopt
  report [map]       grafieken en conclusie opnieuw maken (geen GPU nodig)
  deadman <uren>     doodsklok zetten: stop de pod automatisch
  disarm             doodsklok afzetten
  power-down         nu stoppen

Opties
  --force            doorgaan ondanks waarschuwingen van doctor of /metrics
  --mock             generale repetitie op je laptop tegen de ingebouwde
                     nep-vLLM: geen GPU, geen model, geen kosten
  --deadman <uren>   doodsklok zetten bij 'all'; 'auto' = geschatte duur + 1,5u
  --shutdown         de pod stoppen zodra alles klaar is, ook zonder push
  --no-push          de resultaten niet naar de repo pushen. Dan stopt de pod
                     ook niet vanzelf: de meting staat dan alleen op deze
                     machine, en die mag je niet kwijtraken
  --branch <naam>    branch om de resultaten heen te pushen
                     (standaard resultaten/<kaart>-<tijdstempel>)
  --skip-lesson      fase 2 overslaan. De gecombineerde RESULTATEN.md in de
                     hoofdmap wordt dan alleen bijgewerkt als de vorige
                     lesvalidatie bij deze meting hoort
  --resume-anyway    hervatten in een map die met een ander corpus, gedrags-
                     model of tokenizer gemeten is. Zonder deze vlag stopt de
                     test daarop, want runs met dezelfde naam meten dan iets
                     anders dan wat er al staat
  --skip-engine      de engine-varianten overslaan (die herstarten vLLM)
  --flags "<vlaggen>"  afwijkende vLLM-vlaggen voor 'lesson'
  -c, --config <pad> ander configuratiebestand (standaard config/default.json)
  --set k=v          configuratie overschrijven, wordt doorgegeven aan het harnas
  --detach           zichzelf loskoppelen van de terminal (nohup) en het log tonen

Omgevingsvariabelen
  MODEL SERVED_NAME PORT MAX_MODEL_LEN MAX_NUM_SEQS KV_CACHE_DTYPE GPU_UTIL
  MIN_FREE_GB MIN_CONTAINER_FREE_GB ATTENTION_BACKEND SERVER_START_ATTEMPTS
  TENSOR_PARALLEL VRAM_GB GPU_NAME HF_HOME CONFIG RESULTS_DIR WORKSPACE
  RESULTS_BRANCH PUSH_RETRIES
  HF_TOKEN HF_HUB_OFFLINE HF_DOWNLOAD_RETRIES HF_RATE_LIMIT_WAIT_S
  HF_RATE_LIMIT_MAX_WAIT_S

Voorbeelden
  scripts/pod.sh all --deadman auto
  scripts/pod.sh monitor --label "les 3H woensdag"      # meten tijdens een echte les
  TENSOR_PARALLEL=2 VRAM_GB=64 scripts/pod.sh all      # twee RTX 5090's
  MODEL=Qwen/Qwen2.5-Coder-7B-Instruct MAX_MODEL_LEN=32768 \
    VRAM_GB=24 scripts/pod.sh all --skip-engine         # goedkoop uitproberen
  HF_TOKEN=hf_... scripts/pod.sh setup                  # ruimere limiet bij de Hub
  HF_HUB_OFFLINE=1 scripts/pod.sh serve                # niets van huggingface.co
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
      --no-push) PUSH_RESULTS=0; shift ;;
      --branch) RESULTS_BRANCH="$2"; shift 2 ;;
      --skip-lesson) SKIP_LESSON=1; shift ;;
      --resume-anyway) RESUME_ANYWAY=1; shift ;;
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
    monitor)    cmd_monitor "$@" ;;
    report)     cmd_report "${1:-}" ;;
    deadman)    [ $# -ge 1 ] || die "hoeveel uur?"; mkdir -p "$STATE_DIR"; arm_deadman "$1" ;;
    disarm)     disarm_deadman ;;
    power-down) power_down ;;
    ""|-h|--help) usage ;;
    *) usage; die "onbekend commando: $command" ;;
  esac
}

main "$@"
