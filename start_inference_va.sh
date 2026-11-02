#!/usr/bin/env bash
# One-shot launcher for VLA inference using lingbot-va as the policy backend
# (drop-in replacement for start_inference.sh which uses Isaac-GR00T).
#
# Usage:
#   bash start_inference_va.sh [OPTIONS]
#
# Options:
#   -p, --prompt              Language prompt for the VLA policy
#                             (default: "pick up the apple")
#   -b, --base-model-path     lingbot-va base dir (VAE + T5 + tokenizer)
#                             (default: ${LVA_BASE_PATH:-$HOME/lingbot-va-base})
#   -t, --transformer-path    Fine-tuned transformer checkpoint dir
#                             (default: ${LVA_TRANSFORMER_PATH:-$HOME/lingbot-va/checkpoint_step_5000/transformer})
#   -r, --lingbot-va-repo     Path to the lingbot-va repo (cwd for `python -m wan_va...`)
#                             (default: ${LVA_REPO:-$HOME/lingbot-va})
#   -h, --help                Show this help message
#
# Environment overrides (advanced):
#   LVA_CONDA_ENV       conda env name with lingbot-va installed   (default: lingbot-va)
#   LVA_CONFIG          config-name passed to the adapter          (default: custom)
#   LVA_PORT            ZMQ port for the policy server             (default: 5550)
#   EVICT_T5            1 -> --evict-t5            T5 loads ONCE at startup, encodes the
#                                                  prompt, then is freed forever. Prompt is
#                                                  fixed for the session. (default: 1)
#   OFFLOAD_T5_ONLY     1 -> --offload-t5-only     T5 stays on CPU (~84s prompt encode);
#                                                  supports runtime prompt changes. (default: 0)
#   ENABLE_OFFLOAD      1 -> --enable-offload      T5+VAE both on CPU (slow). (default: 0)
#   NO_KV_WARMUP        1 -> --no-kv-warmup                                    (default: 0)
#   CLOSED_LOOP         1 -> closed-loop KV cache (libero-style; default)
#                       0 -> --no-closed-loop  legacy open-loop (autoregress on own predictions)
#   Precedence          : EVICT_T5 > OFFLOAD_T5_ONLY > ENABLE_OFFLOAD > all-GPU
#   INFERENCE_RATE      VLA forward-pass frequency (Hz)            (default: 2.5)
#   MAX_LATENCY_SKIP    Latency-compensation cap                   (default: 3)
#   TOKEN_EMA_ALPHA     Chunk-boundary EMA smoothing               (default: 1.0)
#   STARTUP_RAMP_HOLD   Pubs to hold anchor (skip chunk-0 LF0)     (default: 20 = 0.4s @50Hz)
#   STARTUP_RAMP_STEPS  Pubs to ramp anchor→model AFTER hold       (default: 60 = 1.2s @50Hz)
#   MASTER_PORT         torch.distributed master port              (default: 29501)
#
# Examples:
#   bash start_inference_va.sh
#   bash start_inference_va.sh --prompt "place the cup on the table"
#   bash start_inference_va.sh -p "open the drawer" \
#       -b /data/lingbot-va-base \
#       -t /data/lingbot-va/checkpoint_step_5000/transformer
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PROMPT="tes"
LVA_BASE_PATH="${LVA_BASE_PATH:-$HOME/lingbot-va-base}"
LVA_TRANSFORMER_PATH="${LVA_TRANSFORMER_PATH:-$HOME/lingbot-va/checkpoint_step_5000/transformer}"
LVA_REPO="${LVA_REPO:-$HOME/lingbot-va}"

LVA_CONDA_ENV="${LVA_CONDA_ENV:-lingbot-va}"
LVA_CONFIG="${LVA_CONFIG:-custom}"
LVA_PORT="${LVA_PORT:-5550}"
EVICT_T5="${EVICT_T5:-1}"
CLOSED_LOOP="${CLOSED_LOOP:-1}"   # 1: libero-style _compute_kv_cache between chunks (default); 0: legacy open-loop
OFFLOAD_T5_ONLY="${OFFLOAD_T5_ONLY:-0}"
ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-0}"
NO_KV_WARMUP="${NO_KV_WARMUP:-0}"

INFERENCE_RATE="${INFERENCE_RATE:-2.5}"
ACTION_HORIZON="${ACTION_HORIZON:-80}"      # lingbot-va emits 4*20=80 actions/chunk; use them all
MAX_LATENCY_SKIP="${MAX_LATENCY_SKIP:-8}"   # tolerate up to 160 ms of inference delay before refresh
TOKEN_EMA_ALPHA="${TOKEN_EMA_ALPHA:-0.4}"   # mild blend at chunk boundary (1.0 = hard snap, 0.0 = freeze)
STARTUP_RAMP_HOLD="${STARTUP_RAMP_HOLD:-20}"    # hold anchor for this many pubs (skips chunk-0 LF0)
STARTUP_RAMP_STEPS="${STARTUP_RAMP_STEPS:-60}"  # publishes blended init→model AFTER hold; HOLD+STEPS=80 @50Hz = 1.6s
export STARTUP_RAMP_HOLD STARTUP_RAMP_STEPS
STARTUP_ANCHOR_FROM_CHUNK="${STARTUP_ANCHOR_FROM_CHUNK:-1}"  # 1 -> anchor startup ramp to model frame-0 (avoids arms-up init-token snap)
export STARTUP_ANCHOR_FROM_CHUNK
MASTER_PORT="${MASTER_PORT:-29501}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prompt)
            PROMPT="$2"; shift 2 ;;
        -b|--base-model-path)
            LVA_BASE_PATH="$2"; shift 2 ;;
        -t|--transformer-path)
            LVA_TRANSFORMER_PATH="$2"; shift 2 ;;
        -r|--lingbot-va-repo)
            LVA_REPO="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown option: $1  (use --help for usage)"; exit 1 ;;
    esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ -d "${LVA_BASE_PATH}" ]] || { echo "ERROR: base-model-path not found: ${LVA_BASE_PATH}"; exit 1; }
[[ -d "${LVA_TRANSFORMER_PATH}" ]] || { echo "ERROR: transformer-path not found: ${LVA_TRANSFORMER_PATH}"; exit 1; }
[[ -d "${LVA_REPO}/wan_va" ]] || { echo "ERROR: lingbot-va repo missing wan_va/: ${LVA_REPO}"; exit 1; }
[[ -f "${LVA_REPO}/wan_va/gr00t_compat_server.py" ]] || {
    echo "ERROR: adapter not found: ${LVA_REPO}/wan_va/gr00t_compat_server.py"; exit 1;
}

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="sonic_inference_va"

cd "$REPO_ROOT"

# Resolve `conda activate` for non-interactive shells.
# Probe common install locations; user can override with CONDA_PROFILE.
CONDA_PROFILE="${CONDA_PROFILE:-}"
if [[ -z "${CONDA_PROFILE}" ]]; then
    for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$c" ]]; then CONDA_PROFILE="$c"; break; fi
    done
fi
[[ -f "${CONDA_PROFILE}" ]] || {
    echo "ERROR: cannot locate conda.sh; set CONDA_PROFILE=/path/to/conda.sh"; exit 1;
}

echo "================================================================"
echo "  SONIC Inference Launcher (lingbot-va backend)"
echo "  Prompt           : ${PROMPT}"
echo "  Base model       : ${LVA_BASE_PATH}"
echo "  Transformer ckpt : ${LVA_TRANSFORMER_PATH}"
echo "  Lingbot-va repo  : ${LVA_REPO}"
echo "  Conda env        : ${LVA_CONDA_ENV}"
echo "  Policy port      : ${LVA_PORT}"
echo "  Inference rate   : ${INFERENCE_RATE} Hz"
echo "  Startup blending : hold ${STARTUP_RAMP_HOLD} + ramp ${STARTUP_RAMP_STEPS} pubs ($(awk "BEGIN{printf \"%.2f\", (${STARTUP_RAMP_HOLD}+${STARTUP_RAMP_STEPS})/50}")s @50Hz)"
echo "  KV cache loop    : $([[ "${CLOSED_LOOP}" == "0" ]] && echo "OPEN (legacy, model autoregresses on own predictions)" || echo "CLOSED (libero-style; real obs fed back each chunk)")"
if   [[ "${EVICT_T5}"        == "1" ]]; then OFFLOAD_DESC="T5 one-shot then freed; VAE+Transformer on GPU (~11GB, prompt fixed)"
elif [[ "${OFFLOAD_T5_ONLY}" == "1" ]]; then OFFLOAD_DESC="T5 on CPU, VAE on GPU (~11GB; 84s per prompt change)"
elif [[ "${ENABLE_OFFLOAD}"  == "1" ]]; then OFFLOAD_DESC="T5+VAE on CPU (~10GB; slow)"
else                                         OFFLOAD_DESC="all on GPU (needs ~23GB)"; fi
echo "  Memory mode      : ${OFFLOAD_DESC}"
echo "================================================================"
echo ""

# Kill any pre-existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Launch the gear_sonic client-side stack ───────────────────────────────────
# It creates and populates the tmux session (camera/keyboard/data-exporter
# panes). The policy server is started by us in a separate window below, so we
# point the client at host=localhost, port=${LVA_PORT}.
python gear_sonic/scripts/launch_inference.py \
    --prompt "${PROMPT}" \
    --camera-host 192.168.123.164 \
    --policy-host localhost \
    --policy-port "${LVA_PORT}" \
    --action-horizon "${ACTION_HORIZON}" \
    --max-latency-skip "${MAX_LATENCY_SKIP}" \
    --token-ema-alpha "${TOKEN_EMA_ALPHA}" \
    --inference-rate "${INFERENCE_RATE}" &

# Wait for the tmux session to appear (up to 30 s)
echo "Waiting for tmux session '${SESSION}'..."
# launch_inference.py creates session 'sonic_inference'; rename it so the two
# stacks (gr00t / lingbot-va) can coexist if desired.
for i in $(seq 1 30); do
    if tmux has-session -t "sonic_inference" 2>/dev/null; then
        tmux rename-session -t "sonic_inference" "$SESSION" 2>/dev/null || true
        echo "  Session '${SESSION}' ready (${i}s)."
        break
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  Session '${SESSION}' ready (${i}s)."
        break
    fi
    sleep 1
done

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session was not created within 30 s. Aborting."
    exit 1
fi

# Give the launcher time to finish populating all four panes
sleep 8

# ── Window: lva-server (lingbot-va via gr00t_compat_server) ──────────────────
OFFLOAD_FLAG=""
if   [[ "${EVICT_T5}"        == "1" ]]; then OFFLOAD_FLAG="--evict-t5"
elif [[ "${OFFLOAD_T5_ONLY}" == "1" ]]; then OFFLOAD_FLAG="--offload-t5-only"
elif [[ "${ENABLE_OFFLOAD}"  == "1" ]]; then OFFLOAD_FLAG="--enable-offload"
fi
KV_FLAG=""
[[ "${NO_KV_WARMUP}" == "1" ]]   && KV_FLAG="--no-kv-warmup"
CL_FLAG=""
[[ "${CLOSED_LOOP}"  == "0" ]]   && CL_FLAG="--no-closed-loop"

# Escape the prompt for safe embedding in a single-line tmux send-keys string.
PROMPT_ESC="${PROMPT//\\/\\\\}"
PROMPT_ESC="${PROMPT_ESC//\"/\\\"}"

tmux new-window -t "$SESSION" -n "lva-server"
tmux send-keys -t "${SESSION}:lva-server" \
    "source ${CONDA_PROFILE} && conda activate ${LVA_CONDA_ENV} && unset VIRTUAL_ENV && \
cd ${LVA_REPO} && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT} \
\${CONDA_PREFIX}/bin/python -u -m wan_va.gr00t_compat_server \
    --config-name ${LVA_CONFIG} \
    --base-model-path ${LVA_BASE_PATH} \
    --transformer-path ${LVA_TRANSFORMER_PATH} \
    --host 0.0.0.0 \
    --port ${LVA_PORT} \
    ${OFFLOAD_FLAG} ${KV_FLAG} ${CL_FLAG} \
    --initial-prompt \"${PROMPT_ESC}\"" C-m

# ── Window: camera-viewer ─────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "camera-viewer"
tmux send-keys -t "${SESSION}:camera-viewer" \
    "cd ${REPO_ROOT} && source .venv_data_collection/bin/activate && \
python gear_sonic/scripts/run_camera_viewer.py \
--camera-host 192.168.123.164 \
--camera-port 5555" C-m

# Focus the main inference window
tmux select-window -t "${SESSION}:inference"

echo ""
echo "================================================================"
echo "  ${SESSION} session ready"
echo ""
echo "  inference     — VLA inference stack (4 panes)"
echo "  lva-server    — lingbot-va policy server (gr00t_compat_server)"
echo "  camera-viewer — live camera feed"
echo ""
echo "  Switch windows : Ctrl+b  n / p"
echo "  Detach         : Ctrl+b  d"
echo "  Kill session   : Ctrl+\\"
echo ""
echo "  Debug tips:"
echo "    - Tail server log : tmux capture-pane -p -t ${SESSION}:lva-server"
echo "    - Dump stack on hang : kill -USR1 <pid of python -m wan_va.gr00t_compat_server>"
echo "================================================================"
echo ""

tmux attach -t "$SESSION"
